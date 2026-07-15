from std.gpu.host import DeviceContext
from std.testing import assert_equal
from layout import TileTensor, row_major
from linalg.gemv import gemv_gpu

def main() raises:
    comptime N = 64
    comptime K = 64
    var ah = alloc[Float32](K)
    var bh = alloc[Float32]((K + 3) * N)
    var ch = alloc[Float32](4 * N)
    for i in range(K):
        ah[i] = 1
    for i in range((K + 3) * N):
        bh[i] = 1
    for i in range(4 * N):
        ch[i] = -7
    with DeviceContext() as ctx:
        var ad = ctx.enqueue_create_buffer[DType.float32](K)
        var bd = ctx.enqueue_create_buffer[DType.float32]((K + 3) * N)
        var cd = ctx.enqueue_create_buffer[DType.float32](4 * N)
        ctx.enqueue_copy(ad, ah)
        ctx.enqueue_copy(bd, bh)
        ctx.enqueue_copy(cd, ch)
        var a = TileTensor(ad, row_major(1, K)).as_immut()
        var b = TileTensor(bd, row_major(K, N)).as_immut()
        var c = TileTensor(cd, row_major(1, N))
        gemv_gpu(c, a, b, ctx)
        ctx.enqueue_copy(ch, cd)
        ctx.synchronize()
    print("C[0] =", ch[0], "; guard C[N] =", ch[N])
    assert_equal(ch[N], Float32(-7), "GEVM wrote beyond its 1xN output")
