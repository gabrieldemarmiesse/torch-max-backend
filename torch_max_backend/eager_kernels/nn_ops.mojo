# ===----------------------------------------------------------------------=== #
# Fast eager-mode NN kernels for max_device: batch norm (inference),
# layer norm, row softmax (with optional causal mask), spatial mean,
# max pool (with indices), embedding gather, and boolean all-reduce.
#
# Same architecture as elementwise_ops.mojo: Python-visible functions get
# `max.driver.Buffer` objects plus the device's DeviceContext pointer, and
# enqueue work on MAX's own device queue (fire and forget, no sync).
#
# Most kernels here are written as a parallel-for over independent output
# elements or rows (`elementwise` with an inner sequential loop), so the same
# code runs on CPU and GPU with fully dynamic shapes. The row-reduction ops
# (layer norm, softmax, argmax/max) additionally have explicit GPU kernels
# that launch one thread block per row: their row counts are far too small
# (batch * seq_len) for a thread-per-row launch to fill the GPU.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.math import sqrt, exp
from std.memory import stack_allocation
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, size_of
from std.utils.coord import Coord
from std.utils.numerics import min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import elementwise

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    FLOAT_DTYPES,
    _enqueue_cached,
    _make_ptr,
    _raw_addr,
    _raw_numel,
    _raw_ctx,
    _raw_dtype,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_f64,
    _raw_tuple_int,
    _raw_tuple_len,
)


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None
](count: Int, ctx: DeviceContext) raises:
    """Run `func` once per index in [0, count) on the device queue."""
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](Coord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Batch norm, inference mode: out = (x - mean[c]) / sqrt(var[c] + eps) * g + b
# Input is NC... contiguous; `inner` is the product of the dims after C.
# ---------------------------------------------------------------------------


@always_inline
def _batch_norm[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    mean_addr: Int,
    var_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    channels: Int,
    inner: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var mean_ptr = _make_ptr[dtype](mean_addr)
    var var_ptr = _make_ptr[dtype](var_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = (i // inner) % channels
        var m = mean_ptr[c].cast[DType.float32]()
        var v = var_ptr[c].cast[DType.float32]()
        var g = gamma_ptr[c].cast[DType.float32]()
        var b = beta_ptr[c].cast[DType.float32]()
        var scale = g / sqrt(v + eps)
        var a = in_ptr[i].cast[DType.float32]()
        out_ptr[i] = ((a - m) * scale + b).cast[dtype]()

    _parallel_for[func](total, ctx)


def _batch_norm_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    mean_buffer: PyObjectPtr,
    var_buffer: PyObjectPtr,
    gamma_buffer: PyObjectPtr,
    beta_buffer: PyObjectPtr,
    params: PyObjectPtr,  # (eps, channels, inner)
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var mean_addr = _raw_addr(mean_buffer)
    var var_addr = _raw_addr(var_buffer)
    var gamma_addr = _raw_addr(gamma_buffer)
    var beta_addr = _raw_addr(beta_buffer)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var channels_val = _raw_tuple_int(params, 1)
    var inner_val = _raw_tuple_int(params, 2)
    var total = _raw_numel(out_buffer)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _batch_norm[dt](
                out_addr,
                in_addr,
                mean_addr,
                var_addr,
                gamma_addr,
                beta_addr,
                eps_val,
                channels_val,
                inner_val,
                total,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast batch_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Layer norm over the last dim; also writes the per-row mean and rstd
# (float32), matching aten.native_layer_norm outputs. The CPU path is one
# parallel task per row. The GPU path launches one thread block per row —
# transformer decode calls this with few rows (batch * seq_len), so a
# thread-per-row launch would leave all but a warp of the GPU idle.
# ---------------------------------------------------------------------------

comptime ROWRED_THREADS = 256
# log2(ROWRED_THREADS): halving steps in the shared-memory reduction trees.
comptime ROWRED_STAGES = 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"layer_norm_block_{dtype}")
def _layer_norm_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    eps: Float32,
    cols: Int,
):
    """One block per row (grid.x = rows); lanes stride over the row and
    tree-reduce the sum and squared-deviation partials in shared memory —
    the same two-pass mean/variance the CPU path computes."""
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += in_ptr[base + j].cast[DType.float32]()
    red[tid] = s
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0] / Float32(cols)
    barrier()
    var mean = bcast[0]

    var vs = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        var d = in_ptr[base + j].cast[DType.float32]() - mean
        vs += d * d
    red[tid] = vs
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        var rstd0 = 1.0 / sqrt(red[0] / Float32(cols) + eps)
        bcast[1] = rstd0
        mean_out_ptr[r] = mean
        rstd_out_ptr[r] = rstd0
    barrier()
    var rstd = bcast[1]

    for j in range(tid, cols, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]()
        var g = gamma_ptr[j].cast[DType.float32]()
        var b = beta_ptr[j].cast[DType.float32]()
        out_ptr[base + j] = ((x - mean) * rstd * g + b).cast[dtype]()


@always_inline
def _layer_norm[
    dtype: DType
](
    out_addr: Int,
    mean_out_addr: Int,
    rstd_out_addr: Int,
    in_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var mean_out_ptr = _make_ptr[DType.float32](mean_out_addr)
    var rstd_out_ptr = _make_ptr[DType.float32](rstd_out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            out_ptr, mean_out_ptr, rstd_out_ptr, in_ptr, gamma_ptr, beta_ptr
        )
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var total = Float32(0)
            for j in range(cols):
                total += in_ptr[base + j].cast[DType.float32]()
            var mean = total / Float32(cols)
            var var_sum = Float32(0)
            for j in range(cols):
                var d = in_ptr[base + j].cast[DType.float32]() - mean
                var_sum += d * d
            var rstd = 1.0 / sqrt(var_sum / Float32(cols) + eps)
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                var g = gamma_ptr[j].cast[DType.float32]()
                var b = beta_ptr[j].cast[DType.float32]()
                out_ptr[base + j] = ((x - mean) * rstd * g + b).cast[dtype]()
            mean_out_ptr[r] = mean
            rstd_out_ptr[r] = rstd

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_layer_norm_block_kernel[dtype]](
                ctx,
                String(t"layer_norm_block_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                mean_out_ptr.as_unsafe_any_origin(),
                rstd_out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                gamma_ptr.as_unsafe_any_origin().as_immutable(),
                beta_ptr.as_unsafe_any_origin().as_immutable(),
                eps,
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _layer_norm_go(
    out_buffer: PyObjectPtr,
    mean_out_buffer: PyObjectPtr,
    rstd_out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    gamma_buffer: PyObjectPtr,
    beta_buffer: PyObjectPtr,
    params: PyObjectPtr,  # (eps, rows, cols)
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var mean_out_addr = _raw_addr(mean_out_buffer)
    var rstd_out_addr = _raw_addr(rstd_out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var gamma_addr = _raw_addr(gamma_buffer)
    var beta_addr = _raw_addr(beta_buffer)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var rows_val = _raw_tuple_int(params, 1)
    var cols_val = _raw_tuple_int(params, 2)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _layer_norm[dt](
                out_addr,
                mean_out_addr,
                rstd_out_addr,
                in_addr,
                gamma_addr,
                beta_addr,
                eps_val,
                rows_val,
                cols_val,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast layer_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise softmax with optional scaling and causal masking, for attention.
# Input is (rows, cols) where rows = batch * q_len. With causal=1, row r
# (query index r % q_len) only attends to columns j <= r % q_len — the
# top-left-aligned tril(ones(L, S)) mask that torch's sdpa is_causal=True
# specifies; masked columns get probability 0.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"softmax_rows_block_{dtype}")
def _softmax_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
):
    """One block per row (grid.x = rows); strided lanes plus shared-memory
    tree reductions for the row max and the exp sum. Same numerics as the
    CPU path: everything in float32, exp evaluated against the true row
    max, one final cast per element."""
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols
    var allowed = cols
    if causal != 0:
        allowed = min(cols, Int(r) % q_len + 1)

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var m = Float32.MIN
    for j in range(tid, allowed, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]() * scale
        if x > m:
            m = x
    red[tid] = m
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            if red[tid + stride] > red[tid]:
                red[tid] = red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0]
    barrier()
    m = bcast[0]

    var s = Float32(0)
    for j in range(tid, allowed, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]() * scale
        s += exp(x - m)
    red[tid] = s
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[1] = red[0]
    barrier()
    var denom = bcast[1]

    for j in range(tid, cols, ROWRED_THREADS):
        if j < allowed:
            var x = in_ptr[base + j].cast[DType.float32]() * scale
            out_ptr[base + j] = (exp(x - m) / denom).cast[dtype]()
        else:
            out_ptr[base + j] = Scalar[dtype](0)


@always_inline
def _softmax_rows[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var allowed = cols
            if causal != 0:
                allowed = min(cols, r % q_len + 1)
            var m = Float32.MIN
            for j in range(allowed):
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                if x > m:
                    m = x
            var denom = Float32(0)
            for j in range(allowed):
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                denom += exp(x - m)
            for j in range(cols):
                if j < allowed:
                    var x = in_ptr[base + j].cast[DType.float32]() * scale
                    out_ptr[base + j] = (exp(x - m) / denom).cast[dtype]()
                else:
                    out_ptr[base + j] = Scalar[dtype](0)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_softmax_rows_block_kernel[dtype]](
                ctx,
                String(t"softmax_rows_block_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                cols,
                scale,
                causal,
                q_len,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _softmax_rows_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    scale: PyObjectPtr,
    causal: PyObjectPtr,
    q_len: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var scale_val = Float32(_raw_f64(scale))
    var causal_val = _raw_int(causal)
    var q_len_val = _raw_int(q_len)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _softmax_rows[dt](
                out_addr,
                in_addr,
                rows_val,
                cols_val,
                scale_val,
                causal_val,
                q_len_val,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast softmax: " + String(dtype))


# ---------------------------------------------------------------------------
# Fused single-query attention (decode step): out = softmax(scale * q @ K^T)
# @ V for q_len == 1, one thread block per (batch * head). Replaces the
# bmm + softmax + bmm chain, whose m=1 GEMMs read K one row per thread
# (uncoalesced) and which costs three kernel launches plus two scratch
# buffers per call. GPU only; the Python side falls back to the generic
# path on CPU or when the size caps below don't hold.
# ---------------------------------------------------------------------------

comptime ATTN_THREADS = 256
comptime ATTN_MAX_KV = 4096
comptime ATTN_MAX_HD = 256


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ATTN_THREADS))
)
@__name(t"attn_decode_{dtype}")
def _attn_decode_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
):
    """q/out are (BH, 1, head_dim); k/v are (BH, kv_len, head_dim), all
    contiguous. Scores for the block's row are staged in shared memory
    (hence the ATTN_MAX_KV cap), softmax uses the same f32 max/sum tree
    reductions as _softmax_rows_block_kernel, and the V pass has lane d
    accumulate output element d so V reads coalesce across lanes."""
    comptime vec_align = 4 * size_of[dtype]()
    var bh = block_idx.x
    var tid = thread_idx.x
    var q_base = bh * head_dim
    var kv_base = bh * kv_len * head_dim

    var q_smem = stack_allocation[
        ATTN_MAX_HD, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var s_smem = stack_allocation[
        ATTN_MAX_KV, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        ATTN_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    for d in range(tid, head_dim, ATTN_THREADS):
        q_smem[d] = q_ptr[q_base + d].cast[DType.float32]()
    barrier()

    var m = Float32.MIN
    for j in range(tid, kv_len, ATTN_THREADS):
        var krow = kv_base + j * head_dim
        var acc = Float32(0)
        for d in range(0, head_dim, 4):
            var k4 = k_ptr.load[width=4, alignment=vec_align](krow + d).cast[
                DType.float32
            ]()
            var q4 = q_smem.load[width=4, alignment=16](d)
            acc += (q4 * k4).reduce_add()
        var s = acc * scale
        s_smem[j] = s
        if s > m:
            m = s
    red[tid] = m
    barrier()
    var stride = ATTN_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            if red[tid + stride] > red[tid]:
                red[tid] = red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0]
    barrier()
    m = bcast[0]

    var s = Float32(0)
    for j in range(tid, kv_len, ATTN_THREADS):
        var e = exp(s_smem[j] - m)
        s_smem[j] = e
        s += e
    red[tid] = s
    barrier()
    stride = ATTN_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[1] = red[0]
    barrier()
    var inv_denom = 1.0 / bcast[1]

    for d in range(tid, head_dim, ATTN_THREADS):
        var acc = Float32(0)
        for j in range(kv_len):
            acc += (
                s_smem[j]
                * v_ptr[kv_base + j * head_dim + d].cast[DType.float32]()
            )
        out_ptr[q_base + d] = (acc * inv_denom).cast[dtype]()


@always_inline
def _attn_decode[
    dtype: DType
](
    out_addr: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    bh: Int,
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_attn_decode_kernel[dtype]](
            ctx,
            String(t"attn_decode_{dtype}"),
            bh,
            1,
            1,
            ATTN_THREADS,
            _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
            _make_ptr[dtype](q_addr).as_unsafe_any_origin().as_immutable(),
            _make_ptr[dtype](k_addr).as_unsafe_any_origin().as_immutable(),
            _make_ptr[dtype](v_addr).as_unsafe_any_origin().as_immutable(),
            kv_len,
            head_dim,
            scale,
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def _attn_decode_go(
    out_buffer: PyObjectPtr,
    q_buffer: PyObjectPtr,
    k_buffer: PyObjectPtr,
    v_buffer: PyObjectPtr,
    # (bh, kv_len, head_dim, scale)
    params: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(q_buffer)
    var out_addr = _raw_addr(out_buffer)
    var q_addr = _raw_addr(q_buffer)
    var k_addr = _raw_addr(k_buffer)
    var v_addr = _raw_addr(v_buffer)
    var bh = _raw_tuple_int(params, 0)
    var kv_len = _raw_tuple_int(params, 1)
    var head_dim = _raw_tuple_int(params, 2)
    var scale = Float32(_raw_tuple_f64(params, 3))
    var ctx = _raw_ctx(device_context_ptr)

    if ctx.api() == "cpu":
        raise Error("fast attn_decode is GPU-only")
    if kv_len > ATTN_MAX_KV or head_dim > ATTN_MAX_HD or head_dim % 4 != 0:
        raise Error("attn_decode size caps violated")

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _attn_decode[dt](
                out_addr,
                q_addr,
                k_addr,
                v_addr,
                bh,
                kv_len,
                head_dim,
                scale,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast attn_decode: " + String(dtype))


# ---------------------------------------------------------------------------
# Mean over the trailing dims: input viewed as (rows, cols), out has `rows`
# elements. Covers aten.mean.dim over the last dims (e.g. global avg pool).
# ---------------------------------------------------------------------------


@always_inline
def _mean_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var r = Int(idx[0].value())
        var base = r * cols
        var total = Float32(0)
        for j in range(cols):
            total += in_ptr[base + j].cast[DType.float32]()
        out_ptr[r] = (total / Float32(cols)).cast[dtype]()

    _parallel_for[func](rows, ctx)


def _mean_rows_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _mean_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast mean: " + String(dtype))


# ---------------------------------------------------------------------------
# Max pool 2D over NCHW contiguous input, with indices (torch semantics:
# index of the max within the flattened H*W input plane, int64).
# `planes` is N * C; one parallel task per output element.
# ---------------------------------------------------------------------------


@always_inline
def _max_pool2d[
    dtype: DType
](
    out_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    dil_h: Int,
    dil_w: Int,
    planes: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, idx_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ow = i % out_w
        var oh = (i // out_w) % out_h
        var plane = i // (out_w * out_h)
        var in_base = plane * in_h * in_w
        var best = min_or_neg_inf[dtype]()
        var best_idx = 0
        for fh in range(kh):
            var ih = oh * stride_h - pad_h + fh * dil_h
            if ih < 0 or ih >= in_h:
                continue
            for fw in range(kw):
                var iw = ow * stride_w - pad_w + fw * dil_w
                if iw < 0 or iw >= in_w:
                    continue
                var v = in_ptr[in_base + ih * in_w + iw]
                if v > best:
                    best = v
                    best_idx = ih * in_w + iw
        out_ptr[i] = best
        idx_ptr[i] = Int64(best_idx)

    _parallel_for[func](planes * out_h * out_w, ctx)


def _max_pool2d_go(
    out_buffer: PyObjectPtr,
    idx_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    params: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var idx_addr = _raw_addr(idx_buffer)
    var in_addr = _raw_addr(in_buffer)
    var in_h = _raw_tuple_int(params, 0)
    var in_w = _raw_tuple_int(params, 1)
    var out_h = _raw_tuple_int(params, 2)
    var out_w = _raw_tuple_int(params, 3)
    var kh = _raw_tuple_int(params, 4)
    var kw = _raw_tuple_int(params, 5)
    var stride_h = _raw_tuple_int(params, 6)
    var stride_w = _raw_tuple_int(params, 7)
    var pad_h = _raw_tuple_int(params, 8)
    var pad_w = _raw_tuple_int(params, 9)
    var dil_h = _raw_tuple_int(params, 10)
    var dil_w = _raw_tuple_int(params, 11)
    var planes = _raw_tuple_int(params, 12)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _max_pool2d[dt](
                out_addr,
                idx_addr,
                in_addr,
                in_h,
                in_w,
                out_h,
                out_w,
                kh,
                kw,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
                planes,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast max_pool2d: " + String(dtype))


# ---------------------------------------------------------------------------
# Embedding lookup: out[i] = weight[indices[i // row_len] * row_len +
# i % row_len]. This is gather along dim 0 of a 2D weight table.
# ---------------------------------------------------------------------------


@always_inline
def _gather0[
    dtype: DType, idx_dtype: DType
](
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var weight_ptr = _make_ptr[dtype](weight_addr)
    var indices_ptr = _make_ptr[idx_dtype](indices_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, weight_ptr, indices_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var row = Int(indices_ptr[i // row_len])
        out_ptr[i] = weight_ptr[row * row_len + i % row_len]

    _parallel_for[func](num_indices * row_len, ctx)


@always_inline
def _gather0_data_dispatch[
    idx_dtype: DType
](
    dtype: DType,
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _gather0[dt, idx_dtype](
                out_addr, weight_addr, indices_addr, num_indices, row_len, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast embedding: " + String(dtype))


def _gather0_go(
    out_buffer: PyObjectPtr,
    weight_buffer: PyObjectPtr,
    indices_buffer: PyObjectPtr,
    num_indices: PyObjectPtr,
    row_len: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(weight_buffer)
    var idx_dtype = _raw_dtype(indices_buffer)
    var out_addr = _raw_addr(out_buffer)
    var weight_addr = _raw_addr(weight_buffer)
    var indices_addr = _raw_addr(indices_buffer)
    var num_indices_val = _raw_int(num_indices)
    var row_len_val = _raw_int(row_len)
    var ctx = _raw_ctx(device_context_ptr)

    if idx_dtype == DType.int64:
        _gather0_data_dispatch[DType.int64](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            ctx,
        )
    elif idx_dtype == DType.int32:
        _gather0_data_dispatch[DType.int32](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            ctx,
        )
    else:
        raise Error(
            "unsupported index dtype for fast embedding: " + String(idx_dtype)
        )


# ---------------------------------------------------------------------------
# all() over a bool tensor -> scalar bool: one 256-thread block, strided
# scan + shared-memory AND-tree (was a single sequential GPU thread, which
# sat on the decode critical path via the HF sdpa mask check).
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name("pure_all_bool_block")
def _all_bool_kernel(
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    size: Int,
):
    var tid = Int(thread_idx.x)
    var ok = True
    for j in range(tid, size, ROWRED_THREADS):
        if not in_ptr[j]:
            ok = False
    var red = stack_allocation[
        ROWRED_THREADS, DType.bool, address_space=AddressSpace.SHARED
    ]()
    red[tid] = ok
    barrier()
    comptime for stage in range(ROWRED_STAGES):
        comptime half = ROWRED_THREADS >> (stage + 1)
        if tid < half:
            red[tid] = red[tid] and red[tid + half]
        barrier()
    if tid == 0:
        out_ptr[0] = red[0]


def _all_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr).as_unsafe_any_origin()
    var in_ptr = (
        _make_ptr[DType.bool](in_addr).as_unsafe_any_origin().as_immutable()
    )
    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var result = True
            for j in range(size):
                if not in_ptr[j]:
                    result = False
                    break
            out_ptr[0] = result

        _parallel_for[func](1, ctx)
        return
    _enqueue_cached[_all_bool_kernel](
        ctx,
        String("all_bool_block"),
        1,
        1,
        1,
        ROWRED_THREADS,
        out_ptr,
        in_ptr,
        size,
    )


def _all_bool_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    size: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    _all_bool(
        _raw_addr(out_buffer),
        _raw_addr(in_buffer),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# any() over a bool tensor -> scalar bool: same block reduction as all().
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name("pure_any_bool_block")
def _any_bool_kernel(
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    size: Int,
):
    var tid = Int(thread_idx.x)
    var found = False
    for j in range(tid, size, ROWRED_THREADS):
        if in_ptr[j]:
            found = True
    var red = stack_allocation[
        ROWRED_THREADS, DType.bool, address_space=AddressSpace.SHARED
    ]()
    red[tid] = found
    barrier()
    comptime for stage in range(ROWRED_STAGES):
        comptime half = ROWRED_THREADS >> (stage + 1)
        if tid < half:
            red[tid] = red[tid] or red[tid + half]
        barrier()
    if tid == 0:
        out_ptr[0] = red[0]


@always_inline
def _any_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr).as_unsafe_any_origin()
    var in_ptr = (
        _make_ptr[DType.bool](in_addr).as_unsafe_any_origin().as_immutable()
    )
    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var result = False
            for j in range(size):
                if in_ptr[j]:
                    result = True
                    break
            out_ptr[0] = result

        _parallel_for[func](1, ctx)
        return
    _enqueue_cached[_any_bool_kernel](
        ctx,
        String("any_bool_block"),
        1,
        1,
        1,
        ROWRED_THREADS,
        out_ptr,
        in_ptr,
        size,
    )


def _any_bool_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    size: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    _any_bool(
        _raw_addr(out_buffer),
        _raw_addr(in_buffer),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# Row-wise argmax: input viewed as (rows, cols), out is `rows` int64
# indices (first occurrence wins, matching torch). Covers argmax over the
# vocab dim in greedy decoding, where rows=1 and cols can be > 50000 —
# a single sequential task per row would leave the GPU almost idle, so the
# GPU path launches one thread block per row and reduces across the row in
# parallel. CPU keeps the original single-task-per-row scalar scan.
# ---------------------------------------------------------------------------

comptime ARGMAX_THREADS = 256
# log2(ARGMAX_THREADS): number of halving steps in the shared-memory
# reduction tree below.
comptime ARGMAX_STAGES = 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ARGMAX_THREADS))
)
@__name(t"argmax_rows_block_{dtype}")
def _argmax_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols: Int,
):
    """One block per row (grid.x = rows); ARGMAX_THREADS lanes each stride
    over the row picking their own best (value, index) with strict `>` (so
    ties keep the lane's earliest index), then a shared-memory tree
    reduction combines lanes with an explicit lower-index tiebreak on equal
    values. Together these preserve torch's first-occurrence-wins argmax
    semantics regardless of how work is split across lanes.
    """
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = min_or_neg_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ARGMAX_THREADS):
        var v = in_ptr[base + j]
        if v > best_val:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ARGMAX_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ARGMAX_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ARGMAX_THREADS // 2
    for _ in range(ARGMAX_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            if other_val > cur_val or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[r] = idx_smem[0]


@always_inline
def _argmax_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                if v > best:
                    best = v
                    best_idx = j
            out_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            var out_p = out_ptr.as_unsafe_any_origin()
            var in_p = in_ptr.as_unsafe_any_origin().as_immutable()
            _enqueue_cached[_argmax_rows_block_kernel[dtype]](
                ctx,
                String(t"argmax_rows_{dtype}"),
                rows,
                1,
                1,
                ARGMAX_THREADS,
                out_p,
                in_p,
                cols,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _argmax_rows_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
    ]:
        if dtype == dt:
            _argmax_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast argmax: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise max reduction (values only): input viewed as (rows, cols), out
# has `rows` elements of the same dtype. rows=1 covers aten.max() (no dim).
# ---------------------------------------------------------------------------


@always_inline
def _max_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var r = Int(idx[0].value())
        var base = r * cols
        var best = in_ptr[base]
        for j in range(1, cols):
            var v = in_ptr[base + j]
            if v > best:
                best = v
        out_ptr[r] = best

    _parallel_for[func](rows, ctx)


def _max_rows_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int64,
        DType.int32,
    ]:
        if dtype == dt:
            _max_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast max: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise cumulative sum along the last dim: input viewed as (rows, cols).
# One sequential task per row — used on the small int tensors of the
# generation loop (position ids from attention-mask cumsum).
# ---------------------------------------------------------------------------


@always_inline
def _cumsum_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var r = Int(idx[0].value())
        var base = r * cols
        var total = Scalar[dtype](0)
        for j in range(cols):
            total += in_ptr[base + j]
            out_ptr[base + j] = total

    _parallel_for[func](rows, ctx)


def _cumsum_rows_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [DType.int64, DType.int32, DType.float32]:
        if dtype == dt:
            _cumsum_rows[dt](out_addr, in_addr, rows_val, cols_val, ctx)
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast cumsum: " + String(dtype))


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _batch_norm_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _batch_norm_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except:
        pass
    return _raw_ret_none()


def _layer_norm_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _layer_norm_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except:
        pass
    return _raw_ret_none()


def _softmax_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _softmax_rows_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except:
        pass
    return _raw_ret_none()


def _attn_decode_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _attn_decode_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _mean_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _mean_rows_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _max_pool2d_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _max_pool2d_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _gather0_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _gather0_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _all_bool_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _all_bool_go(args[0], args[1], args[2], args[3])
    except:
        pass
    return _raw_ret_none()


def _any_bool_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _any_bool_go(args[0], args[1], args[2], args[3])
    except:
        pass
    return _raw_ret_none()


def _argmax_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _argmax_rows_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _max_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _max_rows_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _cumsum_rows_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _cumsum_rows_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_nn_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("nn_ops")
        b.def_py_c_function(
            _batch_norm_dispatcher,
            "BatchNormInference",
            docstring=(
                "out = (x - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]"
                " (NC..., contiguous)"
            ),
        )
        b.def_py_c_function(
            _layer_norm_dispatcher,
            "LayerNorm",
            docstring=(
                "layer norm over the last dim; also writes float32 mean/rstd"
                " per row"
            ),
        )
        b.def_py_c_function(
            _softmax_rows_dispatcher,
            "SoftmaxRows",
            docstring="row softmax of scale*x with optional causal mask",
        )
        b.def_py_c_function(
            _attn_decode_dispatcher,
            "AttnDecode",
            docstring=(
                "fused q_len==1 attention: softmax(scale * q @ K^T) @ V,"
                " one block per batch*head (GPU only)"
            ),
        )
        b.def_py_c_function(
            _mean_rows_dispatcher,
            "MeanRows",
            docstring="mean over the trailing dims (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _max_pool2d_dispatcher,
            "MaxPool2dWithIndices",
            docstring=(
                "max pool over NCHW contiguous input, returns values and int64"
                " plane indices"
            ),
        )
        b.def_py_c_function(
            _gather0_dispatcher,
            "Gather0",
            docstring="embedding lookup: gather rows of a 2D table",
        )
        b.def_py_c_function(
            _all_bool_dispatcher,
            "AllBool",
            docstring="all() over a bool tensor -> scalar bool",
        )
        b.def_py_c_function(
            _any_bool_dispatcher,
            "AnyBool",
            docstring="any() over a bool tensor -> scalar bool",
        )
        b.def_py_c_function(
            _argmax_rows_dispatcher,
            "ArgmaxRows",
            docstring="argmax over the last dim (rows, cols) -> int64 (rows,)",
        )
        b.def_py_c_function(
            _max_rows_dispatcher,
            "MaxRows",
            docstring="max over the last dim (rows, cols) -> (rows,)",
        )
        b.def_py_c_function(
            _cumsum_rows_dispatcher,
            "CumsumRows",
            docstring="cumulative sum along the last dim (rows, cols)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create nn_ops python module: {e}")
