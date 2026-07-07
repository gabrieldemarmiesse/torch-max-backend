# ===----------------------------------------------------------------------=== #
# Fast eager-mode matmul kernels for max_device.
#
# `Matmul` / `Bmm` run pure-Mojo GEMM kernels (shared-memory tiled with
# per-thread register tiles, plus a bandwidth-oriented small-M variant) so
# the fast path works with only the NVIDIA driver — no cuBLAS. All kernels
# accumulate in float32 and handle dynamic shapes with edge guards.
#
# `MatmulVendor` / `BmmVendor` keep the previous vendor BLAS (cuBLAS) path
# available for A/B benchmarking; nothing in the backend calls them.
#
# GPU only: the Python side falls back to the graph path on CPU devices.
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv
from std.memory import stack_allocation
from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from std.gpu.host import DeviceBuffer, DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, size_of
from std.utils.coord import Coord as StdCoord
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import elementwise

from layout import TileTensor, row_major

from linalg.bmm import batched_matmul
from linalg.matmul.vendor.blas import matmul as vendor_matmul

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    FLOAT_DTYPES,
    _enqueue_cached,
    _get_ctx,
    _get_dtype,
    _make_ptr,
    _raw_addr,
    _raw_ctx,
    _raw_dtype,
    _raw_ret_none,
    _raw_tuple_int,
    _raw_tuple_len,
)


# ---------------------------------------------------------------------------
# Tiled GEMM kernel: each block computes a BM x BN tile of C using shared
# memory K-slabs and a TM x TN register tile per thread. Batched via
# block_idx.z. Accumulates in float32.
#
#   C[z, m, n] = A[z, m, k] @ B[z, k, n]     (transpose_b=False)
#   C[z, m, n] = A[z, m, k] @ B[z, n, k]^T   (transpose_b=True)
#
# Split-K: grid.z = batch * ksplits; each split covers a K-chunk and, when
# ksplits > 1, accumulates its partial result into a zero-initialized C
# with fp32 atomics. This keeps enough blocks in flight for the K-rich,
# MN-poor shapes convolution lowers to (e.g. 512x49 @ k=4608).
# ---------------------------------------------------------------------------

# Aim for a few blocks per SM when choosing split-K factors (H100: 114 SMs).
comptime TARGET_BLOCKS = 342


@__name("pure_ksplit_reduce")
def _ksplit_reduce_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mn: Int,
    ksplits: Int,
    total: Int,
):
    # 4 outputs per thread: one div/mod chain + vector loads per chunk.
    var i = (block_idx.x * 256 + thread_idx.x) * 4
    if i >= total:
        return
    var bz = i // mn
    var off = i % mn
    if off + 4 <= mn and i + 4 <= total:
        var base = bz * ksplits * mn + off
        var acc = SIMD[DType.float32, 4](0)
        for st in range(ksplits):
            acc += ws_ptr.load[width=4](base + st * mn)
        out_ptr.store(i, acc)
    else:
        for u in range(4):
            var iu = i + u
            if iu >= total:
                return
            var bzu = iu // mn
            var offu = iu % mn
            var baseu = bzu * ksplits * mn + offu
            var accu = Scalar[DType.float32](0)
            for st in range(ksplits):
                accu += ws_ptr[baseu + st * mn]
            out_ptr[iu] = accu


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDSize(2),
)
@__name(t"pure_gemm_tiled_{dtype}_{BM}x{BN}x{BK}_tb{transpose_b}")
def _gemm_tiled_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime assert (BM * BK) % THREADS == 0, "A tile not evenly loadable"
    comptime assert (BK * BN) % THREADS == 0, "B tile not evenly loadable"
    comptime LA = (BM * BK) // THREADS
    comptime LB = (BK * BN) // THREADS

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    # A slab stored transposed (As[kk][mm]) so the per-thread TM-wide frag
    # load is contiguous; B slab stored as Bs[kk][nn].
    var a_smem = stack_allocation[
        BK * BM, dtype, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        BK * BN, dtype, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[DType.float32, TN], TM](
        fill=SIMD[DType.float32, TN](0)
    )

    for kt in range(k_start, k_end, BK):
        # Cooperative loads, zero-padded on every edge.
        comptime for t in range(LA):
            var i = t * THREADS + tid
            var mm = i // BK
            var kk = i % BK
            var row = bm + mm
            var col = kt + kk
            var val = Scalar[dtype](0)
            if row < m and col < k_end:
                val = a_ptr[row * k + col]
            a_smem[kk * BM + mm] = val

        comptime for t in range(LB):
            var i = t * THREADS + tid
            var val = Scalar[dtype](0)

            comptime if transpose_b:
                var nn = i // BK
                var kk = i % BK
                var row = bn + nn  # row of B (n, k)
                var col = kt + kk
                if row < n and col < k_end:
                    val = b_ptr[row * k + col]
                b_smem[kk * BN + nn] = val
            else:
                var kk = i // BN
                var nn = i % BN
                var row = kt + kk  # row of B (k, n)
                var col = bn + nn
                if row < k_end and col < n:
                    val = b_ptr[row * n + col]
                b_smem[kk * BN + nn] = val

        barrier()

        # Runtime loop on kk: full comptime unrolling of the slab kept ~190
        # live registers (1 block/SM); this stays ~2x lower.
        for kk in range(BK):
            var a_frag = a_smem.load[width=TM](kk * BM + tm0).cast[
                DType.float32
            ]()
            var b_frag = b_smem.load[width=TN](kk * BN + tn0).cast[
                DType.float32
            ]()
            comptime for i in range(TM):
                acc[i] = b_frag.fma(SIMD[DType.float32, TN](a_frag[i]), acc[i])

        barrier()

    comptime for i in range(TM):
        var row = bm + tm0 + i
        if row < m:
            var out = acc[i].cast[dtype]()
            if bn + tn0 + TN <= n:
                c_ptr.store(row * n + bn + tn0, out)
            else:
                comptime for j in range(TN):
                    var col = bn + tn0 + j
                    if col < n:
                        c_ptr[row * n + col] = out[j]


# ---------------------------------------------------------------------------
# Pipelined float32 GEMM kernel: double-buffered shared memory with a
# software pipeline — the next K-slab is fetched (B via cp.async straight
# into shared memory, A staged through registers so it can be stored
# transposed) while the current slab is computed. VEC=4 uses float4 global
# accesses and requires the relevant leading dimension % 4 == 0.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDSize(2 if BM >= 128 else 3),
)
@__name(t"pure_gemm_pipe_{BM}x{BN}x{BK}_va{VEC_A}_vb{VEC_B}_tb{transpose_b}")
def _gemm_pipe_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    VEC_A: Int,
    VEC_B: Int,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime F32 = DType.float32
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime LA = (BM * BK) // THREADS  # A elements per thread per slab
    comptime LB = (BK * BN) // THREADS  # B elements per thread per slab
    comptime NA = LA // VEC_A  # A vectors per thread
    comptime NB = LB // VEC_B  # B vectors per thread
    comptime assert (BM * BK) % (THREADS * VEC_A) == 0
    comptime assert (BK * BN) % (THREADS * VEC_B) == 0

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    # Double-buffered slabs. A is stored transposed (As[kk][mm]) so the
    # compute fragment load is a contiguous vector; B as Bs[kk][nn].
    var a_smem = stack_allocation[
        2 * BK * BM, F32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        2 * BK * BN, F32, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[F32, TN], TM](fill=SIMD[F32, TN](0))
    var a_regs = InlineArray[SIMD[F32, VEC_A], NA](uninitialized=True)
    var b_regs = InlineArray[SIMD[F32, VEC_B], NB](uninitialized=True)

    var nslabs = ceildiv(k_end - k_start, BK)

    # ---- helpers ----------------------------------------------------------

    @always_inline
    @parameter
    @__copy_capture(a_ptr, bm, k_end)
    def _load_a_regs(kt: Int, mut regs: InlineArray[SIMD[F32, VEC_A], NA]):
        # Guarded (row, k) loads, zero-padded past m / k_end.
        comptime for t in range(NA):
            var ci = t * THREADS + tid
            var mm = ci // (BK // VEC_A)
            var ck = (ci % (BK // VEC_A)) * VEC_A
            var row = bm + mm
            var col = kt + ck
            var vec = SIMD[F32, VEC_A](0)
            if row < m:
                if col + VEC_A <= k_end:
                    vec = a_ptr.load[width=VEC_A](row * k + col)
                elif col < k_end:
                    for u in range(k_end - col):
                        vec[u] = a_ptr[row * k + col + u]
            regs[t] = vec

    @always_inline
    @parameter
    @__copy_capture(a_smem, tid)
    def _store_a_smem(buf: Int, regs: InlineArray[SIMD[F32, VEC_A], NA]):
        var base = a_smem + buf * BK * BM
        comptime for t in range(NA):
            var ci = t * THREADS + tid
            var mm = ci // (BK // VEC_A)
            var ck = (ci % (BK // VEC_A)) * VEC_A
            comptime for u in range(VEC_A):
                base[(ck + u) * BM + mm] = regs[t][u]

    @always_inline
    @parameter
    @__copy_capture(b_ptr, bn, n, k, k_end)
    def _cpasync_b(kt: Int, buf: Int):
        # B (k, n) row-major: chunks along n, straight into Bs[kk][nn].
        var base = b_smem + buf * BK * BN
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var kk = ci // (BN // VEC_B)
            var cn = (ci % (BN // VEC_B)) * VEC_B
            var row = kt + kk
            var col = bn + cn
            var bytes: Int32 = 0
            if row < k_end:
                bytes = Int32(max(0, min(VEC_B, n - col)) * 4)
            var src_off = (row * n + col) if bytes > 0 else 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                (base + kk * BN + cn).address_space_cast[AddressSpace.SHARED](),
                src_size=bytes,
            )

    @always_inline
    @parameter
    @__copy_capture(b_ptr, bn, n, k, k_end)
    def _load_b_regs(kt: Int, mut regs: InlineArray[SIMD[F32, VEC_B], NB]):
        # transpose_b: B is (n, k) row-major; chunks along k like A.
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var nn = ci // (BK // VEC_B)
            var ck = (ci % (BK // VEC_B)) * VEC_B
            var row = bn + nn
            var col = kt + ck
            var vec = SIMD[F32, VEC_B](0)
            if row < n:
                if col + VEC_B <= k_end:
                    vec = b_ptr.load[width=VEC_B](row * k + col)
                elif col < k_end:
                    for u in range(k_end - col):
                        vec[u] = b_ptr[row * k + col + u]
            regs[t] = vec

    @always_inline
    @parameter
    @__copy_capture(b_smem, tid)
    def _store_b_smem(buf: Int, regs: InlineArray[SIMD[F32, VEC_B], NB]):
        var base = b_smem + buf * BK * BN
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var nn = ci // (BK // VEC_B)
            var ck = (ci % (BK // VEC_B)) * VEC_B
            comptime for u in range(VEC_B):
                base[(ck + u) * BN + nn] = regs[t][u]

    @always_inline
    @parameter
    def _fetch(kt: Int, buf: Int):
        comptime if transpose_b:
            _load_b_regs(kt, b_regs)
            _store_b_smem(buf, b_regs)
        else:
            _cpasync_b(kt, buf)
            async_copy_commit_group()
        _load_a_regs(kt, a_regs)
        _store_a_smem(buf, a_regs)

    # ---- prologue ---------------------------------------------------------

    if nslabs == 0:
        return
    _fetch(k_start, 0)

    comptime if not transpose_b:
        async_copy_wait_group(0)
    barrier()

    # ---- main loop --------------------------------------------------------

    for s in range(nslabs):
        var cur = s % 2
        var nxt = (s + 1) % 2
        var has_next = s + 1 < nslabs

        # Issue next slab's fetches so they overlap with this slab's math.
        if has_next:
            comptime if not transpose_b:
                _cpasync_b(k_start + (s + 1) * BK, nxt)
                async_copy_commit_group()
            _load_a_regs(k_start + (s + 1) * BK, a_regs)

            comptime if transpose_b:
                _load_b_regs(k_start + (s + 1) * BK, b_regs)

        var a_base_s = a_smem + cur * BK * BM
        var b_base_s = b_smem + cur * BK * BN
        for kk in range(BK):
            var a_frag = a_base_s.load[width=TM](kk * BM + tm0)
            var b_frag = b_base_s.load[width=TN](kk * BN + tn0)
            comptime for i in range(TM):
                acc[i] = b_frag.fma(SIMD[F32, TN](a_frag[i]), acc[i])

        if has_next:
            comptime if not transpose_b:
                async_copy_wait_group(0)
            _store_a_smem(nxt, a_regs)

            comptime if transpose_b:
                _store_b_smem(nxt, b_regs)
        barrier()

    # ---- epilogue ---------------------------------------------------------

    comptime for i in range(TM):
        var row = bm + tm0 + i
        if row < m:
            if bn + tn0 + TN <= n:
                c_ptr.store(row * n + bn + tn0, acc[i])
            else:
                comptime for j in range(TN):
                    var col = bn + tn0 + j
                    if col < n:
                        c_ptr[row * n + col] = acc[i][j]


# ---------------------------------------------------------------------------
# 3-stage cp.async float32 GEMM (transpose_b=False only): both operands are
# fetched straight into shared memory with cp.async, two slabs ahead of the
# compute — enough in-flight distance to cover L2 latency on the L2-resident
# deep-K shapes convolution produces. A is stored row-major (As[mm][kk],
# cp.async cannot transpose); the compute loop reads A as per-row scalar
# broadcasts.
# ---------------------------------------------------------------------------

comptime PIPE3_STAGES = 4


# One thread per element: out (k, m) <- in (m, k). Used to transpose the
# small activation matrix for the skinny-M transposed-B GEMM path below
# (96 KB for a GPT-2 decode step; ~1 µs).
@__name("pure_transpose_small")
def _transpose_small_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    k: Int,
):
    var i = block_idx.x * 256 + thread_idx.x
    if i >= m * k:
        return
    var mm = i // k
    var kk = i % k
    out_ptr[kk * m + mm] = in_ptr[i]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDSize(MINB if MINB > 0 else (2 if BM >= 128 else 3)),
)
@__name(
    t"pure_gemm_pipe3_{BM}x{BN}x{BK}_va{VEC_A}_vb{VEC_B}_ct{CT}_s{STAGES}_at{AT}_mb{MINB}_bs{BIAS}"
)
def _gemm_pipe3_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    VEC_A: Int,
    VEC_B: Int,
    CT: Bool = False,
    STAGES: Int = PIPE3_STAGES,
    AT: Bool = False,
    MINB: Int = -1,
    BIAS: Bool = False,
](
    # MINB: min resident blocks/SM hint (__launch_bounds__ second arg);
    # -1 keeps the historical default (2 for BM >= 128 else 3).
    # BIAS: fuse `c[row, col] += bias[col]` into the epilogue (the ks == 0
    # split only, so split-K partial sums receive the bias exactly once).
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
    bias_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
):
    comptime F32 = DType.float32
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime NA = (BM * BK) // (THREADS * VEC_A)
    comptime NB = (BK * BN) // (THREADS * VEC_B)
    comptime assert (BM * BK) % (THREADS * VEC_A) == 0
    comptime assert (BK * BN) % (THREADS * VEC_B) == 0
    comptime assert not (BIAS and CT), "bias fusion not supported with CT"

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    var a_smem = stack_allocation[
        STAGES * BM * BK, F32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        STAGES * BK * BN, F32, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[F32, TN], TM](fill=SIMD[F32, TN](0))

    var nslabs = ceildiv(k_end - k_start, BK)
    if nslabs == 0:
        return

    @always_inline
    @parameter
    def _fetch(s: Int):
        var buf = s % STAGES
        var kt = k_start + s * BK
        var a_dst = a_smem + buf * BM * BK
        var b_dst = b_smem + buf * BK * BN

        comptime if AT:
            # A^T (k, m) row-major: chunks along m into As[kk][mm], so the
            # compute fragment load is a contiguous vector (no scalar
            # broadcasts). Requires m % VEC_A == 0 for cp.async alignment.
            comptime for t in range(NA):
                var ci = t * THREADS + tid
                var kk = ci // (BM // VEC_A)
                var cm = (ci % (BM // VEC_A)) * VEC_A
                var row = kt + kk
                var col = bm + cm
                var bytes: Int32 = 0
                if row < k_end:
                    bytes = Int32(max(0, min(VEC_A, m - col)) * 4)
                var src_off = (row * m + col) if bytes > 0 else 0
                async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                    (a_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                    (a_dst + kk * BM + cm).address_space_cast[
                        AddressSpace.SHARED
                    ](),
                    src_size=bytes,
                )
        else:
            # A (m, k) row-major: chunks along k into As[mm][kk].
            comptime for t in range(NA):
                var ci = t * THREADS + tid
                var mm = ci // (BK // VEC_A)
                var ck = (ci % (BK // VEC_A)) * VEC_A
                var row = bm + mm
                var col = kt + ck
                var bytes: Int32 = 0
                if row < m:
                    bytes = Int32(max(0, min(VEC_A, k_end - col)) * 4)
                var src_off = (row * k + col) if bytes > 0 else 0
                async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                    (a_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                    (a_dst + mm * BK + ck).address_space_cast[
                        AddressSpace.SHARED
                    ](),
                    src_size=bytes,
                )

        # B (k, n) row-major: chunks along n into Bs[kk][nn].
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var kk = ci // (BN // VEC_B)
            var cn = (ci % (BN // VEC_B)) * VEC_B
            var row = kt + kk
            var col = bn + cn
            var bytes: Int32 = 0
            if row < k_end:
                bytes = Int32(max(0, min(VEC_B, n - col)) * 4)
            var src_off = (row * n + col) if bytes > 0 else 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                (b_dst + kk * BN + cn).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

    # Prologue: fill the pipeline (STAGES - 1 slabs in flight).
    for st in range(min(STAGES - 1, nslabs)):
        _fetch(st)
        async_copy_commit_group()
    for _ in range(nslabs, STAGES - 1):
        async_copy_commit_group()

    for s in range(nslabs):
        # Wait until the group fetched for slab s has landed.
        async_copy_wait_group(Int32(STAGES - 2))
        barrier()

        var buf = s % STAGES
        var a_base_s = a_smem + buf * BM * BK
        var b_base_s = b_smem + buf * BK * BN

        # Fully unrolled with register-double-buffered fragments: the
        # loads for slab column kk + 1 issue before the FMAs of column
        # kk, so the shared-memory load latency hides behind the math
        # instead of stalling every iteration.
        var a_cur: SIMD[F32, TM]
        comptime if AT:
            a_cur = a_base_s.load[width=TM](tm0)
        else:
            a_cur = SIMD[F32, TM](0)
            comptime for i in range(TM):
                a_cur[i] = a_base_s[(tm0 + i) * BK]
        var b_cur = b_base_s.load[width=TN](tn0)
        comptime for kk in range(BK - 1):
            var a_nxt: SIMD[F32, TM]
            comptime if AT:
                a_nxt = a_base_s.load[width=TM]((kk + 1) * BM + tm0)
            else:
                a_nxt = SIMD[F32, TM](0)
                comptime for i in range(TM):
                    a_nxt[i] = a_base_s[(tm0 + i) * BK + kk + 1]
            var b_nxt = b_base_s.load[width=TN]((kk + 1) * BN + tn0)
            comptime for i in range(TM):
                acc[i] = b_cur.fma(SIMD[F32, TN](a_cur[i]), acc[i])
            a_cur = a_nxt
            b_cur = b_nxt
        comptime for i in range(TM):
            acc[i] = b_cur.fma(SIMD[F32, TN](a_cur[i]), acc[i])

        # Release this buffer, then refill it with the slab ahead.
        barrier()
        if s + STAGES - 1 < nslabs:
            _fetch(s + STAGES - 1)
        async_copy_commit_group()

    comptime if CT:
        # C is stored transposed, (n, m) row-major: used by the skinny-M
        # transposed-B path, which computes C^T = B @ A^T with the roles
        # of the operands swapped. Rows (the kernel's m) are contiguous in
        # C, so each column's TM values store as one vector.
        comptime for j in range(TN):
            var col = bn + tn0 + j
            if col < n:
                var row0 = bm + tm0
                var out = SIMD[F32, TM](0)
                comptime for i in range(TM):
                    out[i] = acc[i][j]
                if row0 + TM <= m:
                    c_ptr.store(col * m + row0, out)
                else:
                    comptime for i in range(TM):
                        if row0 + i < m:
                            c_ptr[col * m + row0 + i] = out[i]
    else:
        comptime if BIAS:
            var bias_frag = SIMD[F32, TN](0)
            if ks == 0:
                if bn + tn0 + TN <= n:
                    bias_frag = bias_ptr.load[width=TN](bn + tn0)
                else:
                    comptime for j in range(TN):
                        if bn + tn0 + j < n:
                            bias_frag[j] = bias_ptr[bn + tn0 + j]
            comptime for i in range(TM):
                acc[i] = acc[i] + bias_frag
        comptime for i in range(TM):
            var row = bm + tm0 + i
            if row < m:
                if bn + tn0 + TN <= n:
                    c_ptr.store(row * n + bn + tn0, acc[i])
                else:
                    comptime for j in range(TN):
                        var col = bn + tn0 + j
                        if col < n:
                            c_ptr[row * n + col] = acc[i][j]


# ---------------------------------------------------------------------------
# Small-M GEMM kernel (m <= MR): one thread per output column, streaming B
# at full bandwidth with MR row accumulators in registers. Batched via
# block_idx.z.
# ---------------------------------------------------------------------------

comptime SMALLM_THREADS = 256
comptime SMALLM_MR = 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SMALLM_THREADS)),
    `nvvm.minctasm`=SIMDSize(2),
)
@__name(t"pure_gemm_smallm_{dtype}_tb{transpose_b}")
def _gemm_smallm_kernel[
    dtype: DType,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    var col = block_idx.x * SMALLM_THREADS + thread_idx.x
    if col >= n:
        return

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    var kchunk = ceildiv(k, ksplits)
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var acc = InlineArray[Scalar[DType.float32], SMALLM_MR](
        fill=Scalar[DType.float32](0)
    )

    # Unroll k by 4 so each thread keeps several B loads in flight — the
    # kernel is B-bandwidth-bound and often runs at low occupancy.
    comptime KU = 4
    var k4 = k_start + ((k_end - k_start) // KU) * KU

    @always_inline
    @parameter
    def _load_b(kk: Int) -> Scalar[DType.float32]:
        comptime if transpose_b:
            return b_ptr[col * k + kk].cast[DType.float32]()
        else:
            return b_ptr[kk * n + col].cast[DType.float32]()

    for kt in range(k_start, k4, KU):
        var bv = InlineArray[Scalar[DType.float32], KU](uninitialized=True)
        comptime for u in range(KU):
            bv[u] = _load_b(kt + u)
        comptime for u in range(KU):
            comptime for r in range(SMALLM_MR):
                if r < m:
                    acc[r] = (
                        a_ptr[r * k + kt + u]
                        .cast[DType.float32]()
                        .fma(bv[u], acc[r])
                    )

    for kk in range(k4, k_end):
        var bv = _load_b(kk)
        comptime for r in range(SMALLM_MR):
            if r < m:
                acc[r] = a_ptr[r * k + kk].cast[DType.float32]().fma(bv, acc[r])

    comptime for r in range(SMALLM_MR):
        if r < m:
            c_ptr[r * n + col] = acc[r].cast[dtype]()


# ---------------------------------------------------------------------------
# Dispatch: pick a kernel/config from the runtime shape.
# ---------------------------------------------------------------------------


@always_inline
def _enqueue_pipe[
    BM: Int, BN: Int, BK: Int, TM: Int, TN: Int, transpose_b: Bool
](
    ctx: DeviceContext,
    gx: Int,
    gy: Int,
    gz: Int,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
) raises:
    comptime THREADS = (BM // TM) * (BN // TN)
    var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    var va4 = k % 4 == 0  # A (and B when transposed) is loaded along k

    comptime if transpose_b:
        if va4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, True]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v44_tb1"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        else:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 1, True]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v11_tb1"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
    elif BM >= 128:
        # Compute-bound fat tiles: 2-stage with transposed-A vector frags.
        var vb4 = n % 4 == 0  # B is loaded along n
        if va4 and vb4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v44_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif va4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 1, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v41_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif vb4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 4, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v14_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        else:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 1, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v11_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
    else:
        # Latency-bound thin tiles: 3-stage cp.async pipeline.
        var vb4 = n % 4 == 0  # B is loaded along n
        if va4 and vb4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 4, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v44_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        elif va4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 1, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v41_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        elif vb4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 1, 4, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v14_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        else:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 1, 1, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v11_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )


@always_inline
def _gemm_enqueue[
    dtype: DType, transpose_b: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        var a = _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable()
        var b = _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable()

        # Skinny-M float32 paths (decode-step GEMMs: m = batch <= 32). The
        # BM=64 tiles waste half of every block there; the BM=32 pipe3 tile
        # (128-thread blocks, k-chunk floor 96) beats cuBLAS SGEMM on the
        # GPT-2 decode shapes. For transposed-B (linear/lm_head), compute
        # C^T = B @ A^T instead so B streams via cp.async at full rate.
        comptime if dtype == DType.float32:
            comptime if transpose_b:
                # Fat C^T path for medium-m transposed-B (lm_head at large
                # batch): batch==1 only, grid must fill the GPU on its own.
                if (
                    batch == 1
                    and m > 32
                    and k % 4 == 0
                    and m % 4 == 0
                    and ceildiv(n, 64) * ceildiv(m, 64) >= 342
                ):
                    _ct_enqueue[64, 64, 16, 8, 8, 3, 6](
                        c_addr, a_addr, b_addr, m, n, k, ctx
                    )
                    return
            if m > SMALLM_MR and m <= 32:
                comptime if transpose_b:
                    # CT path: batch==1 only (no batched A^T scratch), and
                    # only when the weight-row grid alone fills the GPU.
                    if (
                        batch == 1
                        and k % 4 == 0
                        and m % 4 == 0
                        and ceildiv(n, 64) >= 128
                    ):
                        _ct_enqueue[64, 32, 32, 4, 4, 3](
                            c_addr, a_addr, b_addr, m, n, k, ctx
                        )
                        return
                    if k % 4 == 0:
                        _tune_enqueue[32, 64, 16, 4, 4, True, False](
                            c_addr,
                            a_addr,
                            b_addr,
                            batch,
                            m,
                            n,
                            k,
                            a_bstride,
                            96,
                            ctx,
                        )
                        return
                else:
                    if k % 4 == 0 and n % 4 == 0:
                        _tune_enqueue[32, 64, 16, 4, 4, False, True](
                            c_addr,
                            a_addr,
                            b_addr,
                            batch,
                            m,
                            n,
                            k,
                            a_bstride,
                            96,
                            ctx,
                        )
                        return

        # Choose the kernel/tile config, then a split-K factor that brings
        # the block count up to a few per SM (float32 only).
        var gx: Int
        var gy: Int
        var kcap: Int

        var use_t128 = False
        if m > SMALLM_MR and m >= 96 and n >= 96:
            # Only use the fat tile when its grid alone fills a wave;
            # otherwise the thinner tiles' extra blocks hide more latency.
            use_t128 = ceildiv(n, 128) * ceildiv(m, 128) * batch >= 114
        var use_n32 = (
            m > SMALLM_MR
            and not use_t128
            and n <= 96
            and dtype == DType.float32
        )

        if m <= SMALLM_MR:
            gx = ceildiv(n, SMALLM_THREADS)
            gy = 1
            kcap = ceildiv(k, 64)
        elif use_t128:
            gx = ceildiv(n, 128)
            gy = ceildiv(m, 128)
            kcap = ceildiv(k, 64)
        elif use_n32:
            gx = ceildiv(n, 32)
            gy = ceildiv(m, 64)
            kcap = ceildiv(k, 192)
        else:
            gx = ceildiv(n, 64)
            gy = ceildiv(m, 64)
            kcap = ceildiv(k, 192)

        var ksplits = 1
        if dtype == DType.float32:
            var base = gx * gy * batch
            if base < TARGET_BLOCKS // 2:
                ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)

        # Split-K partials go to a stream-ordered workspace and a final
        # reduce kernel sums them into C (plain stores, deterministic).
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](
                batch * ksplits * m * n
            )
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[dtype](c_target).as_unsafe_any_origin()

        if m <= SMALLM_MR:
            _enqueue_cached[_gemm_smallm_kernel[dtype, transpose_b]](
                ctx,
                String(t"gemm_smallm_{dtype}_tb{transpose_b}"),
                gx,
                gy,
                batch * ksplits,
                SMALLM_THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif use_t128:
            comptime if dtype == DType.float32:
                _enqueue_pipe[128, 128, 16, 8, 8, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
            else:
                _enqueue_cached[
                    _gemm_tiled_kernel[dtype, 128, 128, 16, 8, 8, transpose_b]
                ](
                    ctx,
                    String(t"gemm_t128_{dtype}_tb{transpose_b}"),
                    gx,
                    gy,
                    batch * ksplits,
                    256,
                    c,
                    a,
                    b,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        elif use_n32:
            comptime if dtype == DType.float32:
                _enqueue_pipe[64, 32, 16, 4, 4, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        else:
            comptime if dtype == DType.float32:
                _enqueue_pipe[64, 64, 16, 4, 4, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
            else:
                _enqueue_cached[
                    _gemm_tiled_kernel[dtype, 64, 64, 16, 4, 4, transpose_b]
                ](
                    ctx,
                    String(t"gemm_t64_{dtype}_tb{transpose_b}"),
                    gx,
                    gy,
                    batch * ksplits,
                    256,
                    c,
                    a,
                    b,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        if ksplits > 1:
            var total = batch * m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        # Keep the workspace alive until its free is enqueued after the
        # reduce (stream-ordered).
        _ = ws^
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _gemm_transb_dispatch[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    if transpose_b != 0:
        _gemm_enqueue[dtype, True](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )
    else:
        _gemm_enqueue[dtype, False](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )


# ---------------------------------------------------------------------------
# Tuning entry point (float32 only): run one GEMM with an explicit tile
# config + split-K chunk floor, selected at runtime. Used by offline sweeps
# to pick the dispatch configs; not called by the backend.
# ---------------------------------------------------------------------------


@always_inline
def _tune_enqueue[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    transpose_b: Bool,
    pipe3: Bool,
    STAGES: Int = PIPE3_STAGES,
    MINB: Int = -1,
    BIAS: Bool = False,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    kchunk_min: Int,
    ctx: DeviceContext,
    bias_addr: Int = 0,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var ksplits = 1
        var base = gx * gy * batch
        var kcap = ceildiv(k, kchunk_min)
        if base < TARGET_BLOCKS // 2:
            ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](
                batch * ksplits * m * n
            )
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[DType.float32](c_target).as_unsafe_any_origin()
        var a = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var b = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        comptime if pipe3:
            var bias = (
                _make_ptr[DType.float32](
                    bias_addr if bias_addr != 0 else a_addr
                )
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 4, False, STAGES, False, MINB, BIAS
                ]
            ](
                ctx,
                String(
                    t"tune_p3_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}_bs{BIAS}"
                ),
                gx,
                gy,
                batch * ksplits,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                bias,
            )
        else:
            _enqueue_cached[
                _gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, transpose_b]
            ](
                ctx,
                String(t"tune_p_{BM}x{BN}x{BK}_{TM}{TN}_tb{transpose_b}"),
                gx,
                gy,
                batch * ksplits,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        if ksplits > 1:
            var total = batch * m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        _ = ws^
    else:
        raise Error("no GPU accelerator available at compile time")


# Skinny-M transposed-B GEMM as C^T = B @ A^T: materialize A^T (m x k ->
# k x m, one tiny kernel), then run the cp.async tb0 pipeline with the
# operand roles swapped — B (n, k) row-major streams as the "A" operand at
# full bandwidth — and the CT epilogue writes C (m, n) directly.
@always_inline
def _ct_enqueue[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    STAGES: Int = PIPE3_STAGES,
    MINB: Int = -1,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var at_buf = ctx.enqueue_create_buffer[DType.float32](m * k)
        var at_addr = Int(at_buf.unsafe_ptr())
        var a_in = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at_out = _make_ptr[DType.float32](at_addr).as_unsafe_any_origin()
        _enqueue_cached[_transpose_small_kernel](
            ctx,
            String("transpose_small"),
            ceildiv(m * k, 256),
            1,
            1,
            256,
            at_out,
            a_in,
            m,
            k,
        )
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        var wa = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at = (
            _make_ptr[DType.float32](at_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        # Swapped roles: kernel-m = n (weight rows), kernel-n = m.
        _enqueue_cached[
            _gemm_pipe3_kernel[
                BM, BN, BK, TM, TN, 4, 4, True, STAGES, False, MINB
            ]
        ](
            ctx,
            String(t"gemm_ct_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}"),
            ceildiv(m, BN),
            ceildiv(n, BM),
            1,
            THREADS,
            c,
            wa,
            at,
            n,
            m,
            k,
            n * k,
            1,
            at,
        )
        _ = at_buf^
    else:
        raise Error("no GPU accelerator available at compile time")


# Medium-M FFMA-dense path: transpose the (small) activation A once so
# BOTH operand slabs stream via cp.async and both compute fragments are
# contiguous vector loads (CUTLASS-SIMT-style instruction density).
@always_inline
def _pipe3t_enqueue[
    BM: Int, BN: Int, BK: Int, TM: Int, TN: Int, STAGES: Int, MINB: Int = -1
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    kchunk_min: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var at_buf = ctx.enqueue_create_buffer[DType.float32](m * k)
        var at_addr = Int(at_buf.unsafe_ptr())
        var a_in = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at_out = _make_ptr[DType.float32](at_addr).as_unsafe_any_origin()
        _enqueue_cached[_transpose_small_kernel](
            ctx,
            String("transpose_small"),
            ceildiv(m * k, 256),
            1,
            1,
            256,
            at_out,
            a_in,
            m,
            k,
        )
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var ksplits = 1
        var base = gx * gy
        var kcap = ceildiv(k, kchunk_min)
        if base < TARGET_BLOCKS // 2:
            ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](ksplits * m * n)
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[DType.float32](c_target).as_unsafe_any_origin()
        var at = (
            _make_ptr[DType.float32](at_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var b = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        _enqueue_cached[
            _gemm_pipe3_kernel[
                BM, BN, BK, TM, TN, 4, 4, False, STAGES, True, MINB
            ]
        ](
            ctx,
            String(t"gemm_p3t_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}"),
            gx,
            gy,
            ksplits,
            THREADS,
            c,
            at,
            b,
            m,
            n,
            k,
            m * k,
            ksplits,
            at,
        )
        if ksplits > 1:
            var total = m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        _ = ws^
        _ = at_buf^
    else:
        raise Error("no GPU accelerator available at compile time")


def _matmul_tune_dispatcher(
    c_buffer: PythonObject,
    a_buffer: PythonObject,
    b_buffer: PythonObject,
    # (m, n, k, transpose_b, cfg, kchunk_min); float32 only, k % 4 == 0
    # and (n % 4 == 0 when transpose_b == 0) required.
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var c_addr = Int(py=c_buffer._data_ptr())
    var a_addr = Int(py=a_buffer._data_ptr())
    var b_addr = Int(py=b_buffer._data_ptr())
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var transpose_b = Int(py=params[3])
    var cfg = Int(py=params[4])
    var kmin = Int(py=params[5])
    var ctx = _get_ctx(device_context_ptr)

    if transpose_b == 0:
        if cfg == 1:
            _tune_enqueue[32, 64, 16, 4, 4, False, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 2:
            _tune_enqueue[64, 64, 16, 4, 4, False, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 3:
            _tune_enqueue[128, 128, 16, 8, 8, False, False](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 4:
            _tune_enqueue[128, 64, 8, 8, 8, False, True, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 5:
            _tune_enqueue[128, 128, 8, 8, 8, False, True, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 6:
            _tune_enqueue[128, 64, 8, 8, 8, False, True, 7](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 7:
            _pipe3t_enqueue[128, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 8:
            _pipe3t_enqueue[128, 128, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 9:
            _pipe3t_enqueue[128, 64, 16, 8, 8, 3](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 10:
            _pipe3t_enqueue[64, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 11:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, 4, 4](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 12:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, 4, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 13:
            _pipe3t_enqueue[64, 64, 8, 8, 4, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 14:
            _pipe3t_enqueue[64, 64, 8, 8, 4, 6, 8](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 15:
            _pipe3t_enqueue[128, 64, 8, 8, 8, 5, 4](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 16:
            _pipe3t_enqueue[64, 128, 8, 8, 8, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        else:
            raise Error("unknown tb0 tune cfg")
    else:
        if cfg == 1:
            _tune_enqueue[32, 64, 16, 4, 4, True, False](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 2:
            _ct_enqueue[64, 32, 32, 4, 4, 3](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 3:
            _ct_enqueue[64, 32, 16, 4, 4](c_addr, a_addr, b_addr, m, n, k, ctx)
        elif cfg == 4:
            _ct_enqueue[128, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 5:
            _ct_enqueue[128, 128, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 6:
            _ct_enqueue[64, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 7:
            _ct_enqueue[64, 64, 8, 8, 8, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 8:
            _ct_enqueue[64, 128, 8, 8, 8, 5, 4](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 9:
            _ct_enqueue[64, 64, 16, 8, 8, 3, 6](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        else:
            raise Error("unknown tb1 tune cfg")


@always_inline
def _gemm_dtype_dispatch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Int,
    c_off: Int,  # element offsets into c/a/b
    a_off: Int,
    b_off: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            # c_off/a_off/b_off are element offsets; scale to bytes for the
            # raw addresses (4 for float32, 2 for the 16-bit dtypes).
            comptime elem_size = size_of[dt]()
            _gemm_transb_dispatch[dt](
                c_addr + c_off * elem_size,
                a_addr + a_off * elem_size,
                b_addr + b_off * elem_size,
                batch,
                m,
                n,
                k,
                a_bstride,
                transpose_b,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast matmul: " + String(dtype))


def _matmul_go(
    c_buffer: PyObjectPtr,
    a_buffer: PyObjectPtr,
    b_buffer: PyObjectPtr,
    # (m, n, k, transpose_b) or, with element offsets into the three
    # buffers (grouped convolution), (m, n, k, transpose_b, c_off, a_off,
    # b_off).
    params: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(a_buffer)
    var c_addr = _raw_addr(c_buffer)
    var a_addr = _raw_addr(a_buffer)
    var b_addr = _raw_addr(b_buffer)
    var m = _raw_tuple_int(params, 0)
    var n = _raw_tuple_int(params, 1)
    var k = _raw_tuple_int(params, 2)
    var transpose_b = _raw_tuple_int(params, 3)
    var c_off = 0
    var a_off = 0
    var b_off = 0
    if _raw_tuple_len(params) > 4:
        c_off = _raw_tuple_int(params, 4)
        a_off = _raw_tuple_int(params, 5)
        b_off = _raw_tuple_int(params, 6)
    var ctx = _raw_ctx(device_context_ptr)

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        1,
        m,
        n,
        k,
        m * k,
        transpose_b,
        c_off,
        a_off,
        b_off,
        ctx,
    )


def _bmm_go(
    c_buffer: PyObjectPtr,
    a_buffer: PyObjectPtr,
    b_buffer: PyObjectPtr,
    # (batch, m, n, k, transpose_b) or (batch, m, n, k, transpose_b,
    # a_shared) — a_shared=1 broadcasts a single (m, k) A across the batch
    # (batched convolution with shared weights).
    params: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(a_buffer)
    var c_addr = _raw_addr(c_buffer)
    var a_addr = _raw_addr(a_buffer)
    var b_addr = _raw_addr(b_buffer)
    var batch = _raw_tuple_int(params, 0)
    var m = _raw_tuple_int(params, 1)
    var n = _raw_tuple_int(params, 2)
    var k = _raw_tuple_int(params, 3)
    var transpose_b = _raw_tuple_int(params, 4)
    var a_bstride = m * k
    if _raw_tuple_len(params) > 5 and _raw_tuple_int(params, 5) != 0:
        a_bstride = 0
    var ctx = _raw_ctx(device_context_ptr)

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        batch,
        m,
        n,
        k,
        a_bstride,
        transpose_b,
        0,
        0,
        0,
        ctx,
    )


# ---------------------------------------------------------------------------
# Vendor BLAS (cuBLAS) reference path — benchmarking only, never called by
# the backend. Calling these loads the vendor library into the process.
# ---------------------------------------------------------------------------


@always_inline
def _matmul_vendor[
    dtype: DType, transpose_b: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        # use_tf32=False: full-precision fp32 GEMM, matching torch's CUDA
        # matmul default. The higher-level dispatchers hardcode TF32 on.
        var c = TileTensor(_make_ptr[dtype](c_addr), row_major(m, n))
        var a = TileTensor(_make_ptr[dtype](a_addr), row_major(m, k))
        comptime if transpose_b:
            var b = TileTensor(_make_ptr[dtype](b_addr), row_major(n, k))
            vendor_matmul(ctx, c, a, b, c_row_major=True, transpose_b=True)
        else:
            var b = TileTensor(_make_ptr[dtype](b_addr), row_major(k, n))
            vendor_matmul(ctx, c, a, b, c_row_major=True, transpose_b=False)
    else:
        raise Error("no GPU accelerator available at compile time")


def _matmul_vendor_dispatcher(
    c_buffer: PythonObject,
    a_buffer: PythonObject,
    b_buffer: PythonObject,
    params: PythonObject,  # (m, n, k, transpose_b)
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(a_buffer)
    var c_addr = Int(py=c_buffer._data_ptr())
    var a_addr = Int(py=a_buffer._data_ptr())
    var b_addr = Int(py=b_buffer._data_ptr())
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var transpose_b = Int(py=params[3])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        if transpose_b != 0:
            _matmul_vendor[DType.float32, True](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        else:
            _matmul_vendor[DType.float32, False](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
    else:
        raise Error("vendor matmul benchmark path only supports float32")


@always_inline
def _bmm_vendor[
    dtype: DType, transpose_b: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        var c = TileTensor(_make_ptr[dtype](c_addr), row_major(batch, m, n))
        var a = TileTensor(_make_ptr[dtype](a_addr), row_major(batch, m, k))
        comptime if transpose_b:
            var b = TileTensor(_make_ptr[dtype](b_addr), row_major(batch, n, k))
            batched_matmul[transpose_b=True, target="gpu"](c, a, b, context=ctx)
        else:
            var b = TileTensor(_make_ptr[dtype](b_addr), row_major(batch, k, n))
            batched_matmul[transpose_b=False, target="gpu"](
                c, a, b, context=ctx
            )
    else:
        raise Error("no GPU accelerator available at compile time")


def _bmm_vendor_dispatcher(
    c_buffer: PythonObject,
    a_buffer: PythonObject,
    b_buffer: PythonObject,
    params: PythonObject,  # (batch, m, n, k, transpose_b)
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(a_buffer)
    var c_addr = Int(py=c_buffer._data_ptr())
    var a_addr = Int(py=a_buffer._data_ptr())
    var b_addr = Int(py=b_buffer._data_ptr())
    var batch = Int(py=params[0])
    var m = Int(py=params[1])
    var n = Int(py=params[2])
    var k = Int(py=params[3])
    var transpose_b = Int(py=params[4])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        if transpose_b != 0:
            _bmm_vendor[DType.float32, True](
                c_addr, a_addr, b_addr, batch, m, n, k, ctx
            )
        else:
            _bmm_vendor[DType.float32, False](
                c_addr, a_addr, b_addr, batch, m, n, k, ctx
            )
    else:
        raise Error("vendor bmm benchmark path only supports float32")


# ---------------------------------------------------------------------------
# In-place row-broadcast bias add: out[i] += bias[i % cols]. Used as the
# addmm / conv-bias epilogue.
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_row[
    dtype: DType
](
    out_addr: Int, bias_addr: Int, total: Int, cols: Int, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[i] = out_ptr[i] + bias_ptr[i % cols]

    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](StdCoord(total), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](StdCoord(total), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _bias_add_row_dispatcher(
    out_buffer: PythonObject,
    bias_buffer: PythonObject,
    cols: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(out_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var bias_addr = Int(py=bias_buffer._data_ptr())
    var total = Int(py=out_buffer.num_elements)
    var cols_val = Int(py=cols)
    var ctx = _get_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _bias_add_row[dt](out_addr, bias_addr, total, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


def _matmul_bias_go(
    c_buffer: PyObjectPtr,
    a_buffer: PyObjectPtr,
    b_buffer: PyObjectPtr,
    bias_buffer: PyObjectPtr,
    # (m, n, k, transpose_b)
    params: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    """C = A @ B (+ bias broadcast over rows) in one Python-visible call.

    Equivalent to Matmul followed by BiasAddRow; fused at the dispatcher
    level because addmm/linear call both back to back on every transformer
    layer and each Python->Mojo binding call costs a few microseconds of
    argument unwrapping.
    """
    var dtype = _raw_dtype(a_buffer)
    var c_addr = _raw_addr(c_buffer)
    var a_addr = _raw_addr(a_buffer)
    var b_addr = _raw_addr(b_buffer)
    var bias_addr = _raw_addr(bias_buffer)
    var m = _raw_tuple_int(params, 0)
    var n = _raw_tuple_int(params, 1)
    var k = _raw_tuple_int(params, 2)
    var transpose_b = _raw_tuple_int(params, 3)
    var ctx = _raw_ctx(device_context_ptr)

    # Fused-epilogue path: the float32 pipe3 tile kernels add the bias in
    # the GEMM epilogue, saving a full extra read+write of C per call.
    if (
        dtype == DType.float32
        and transpose_b == 0
        and m > SMALLM_MR
        and k % 4 == 0
        and n % 4 == 0
    ):
        if m <= 32:
            _tune_enqueue[
                32, 64, 16, 4, 4, False, True, PIPE3_STAGES, -1, True
            ](c_addr, a_addr, b_addr, 1, m, n, k, m * k, 96, ctx, bias_addr)
        else:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, PIPE3_STAGES, 4, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, 192, ctx, bias_addr
            )
        return

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        1,
        m,
        n,
        k,
        m * k,
        transpose_b,
        0,
        0,
        0,
        ctx,
    )
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _bias_add_row[dt](c_addr, bias_addr, m * n, n, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


# METH_FASTCALL wrappers for the hot dispatchers (raw CPython unpack; the
# internal Python callers guarantee argument types, and raise sites are
# unsupported-dtype guards gated upstream).


def _matmul_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _matmul_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _bmm_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _bmm_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _matmul_bias_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _matmul_bias_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_matmul_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("matmul_ops")
        b.def_py_c_function(
            _matmul_dispatcher,
            "Matmul",
            docstring=(
                "C = A @ B (row-major, optional transposed B), pure Mojo"
                " kernels, GPU only"
            ),
        )
        b.def_py_c_function(
            _matmul_bias_dispatcher,
            "MatmulBias",
            docstring=(
                "C = A @ B + bias (bias broadcast over rows); one call"
                " instead of Matmul + BiasAddRow"
            ),
        )
        b.def_py_c_function(
            _bmm_dispatcher,
            "Bmm",
            docstring=(
                "batched C = A @ B (rank 3, optional transposed B), pure Mojo"
                " kernels, GPU only"
            ),
        )
        b.def_function[_matmul_vendor_dispatcher](
            "MatmulVendor",
            docstring="vendor BLAS (cuBLAS) matmul — benchmarking only",
        )
        b.def_function[_matmul_tune_dispatcher](
            "MatmulTune",
            docstring=(
                "GEMM with explicit tile cfg + split-K floor — tuning only"
            ),
        )
        b.def_function[_bmm_vendor_dispatcher](
            "BmmVendor",
            docstring="vendor BLAS batched matmul — benchmarking only",
        )
        b.def_function[_bias_add_row_dispatcher](
            "BiasAddRow", docstring="in-place out[i] += bias[i % cols]"
        )
        return b.finalize()
    except e:
        abort(t"failed to create matmul_ops python module: {e}")
