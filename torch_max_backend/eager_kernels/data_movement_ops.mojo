# ===----------------------------------------------------------------------=== #
# Fast eager-mode data-movement kernels for max_device: strided permute
# copies (transpose/permute materialization), narrow copies (split/slice
# along one dim), and dtype casts.
#
# Same architecture as elementwise_ops.mojo: Python-visible functions get
# `max.driver.Buffer` objects plus the device's DeviceContext pointer, and
# enqueue work on MAX's own device queue (fire and forget, no sync).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, size_of
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
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
def _element_size(dtype: DType) raises -> Int:
    """Element size in bytes; copies dispatch on size, not dtype."""
    if dtype == DType.float32 or dtype == DType.int32 or dtype == DType.uint32:
        return 4
    if (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.int16
        or dtype == DType.uint16
    ):
        return 2
    if dtype == DType.float64 or dtype == DType.int64 or dtype == DType.uint64:
        return 8
    if dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 1
    raise Error("unsupported dtype for fast copy: " + String(dtype))


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None
](count: Int, ctx: DeviceContext) raises:
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](Coord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Permute copy: materialize an arbitrary permutation of a contiguous tensor
# of rank <= 4. The Python side pads the *output* shape to 4 dims with
# leading 1s and passes, for each output dim, the corresponding stride in
# the *source* buffer (in elements). Kernels are specialized on element
# byte-size, not dtype, since this is a pure copy.
# ---------------------------------------------------------------------------


@always_inline
def _permute_copy[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    s0: Int,
    s1: Int,
    s2: Int,
    s3: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        out_ptr[i] = in_ptr[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]

    _parallel_for[func](d0 * d1 * d2 * d3, ctx)


def _permute_copy_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    dims: PyObjectPtr,
    strides: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var d0 = _raw_tuple_int(dims, 0)
    var d1 = _raw_tuple_int(dims, 1)
    var d2 = _raw_tuple_int(dims, 2)
    var d3 = _raw_tuple_int(dims, 3)
    var s0 = _raw_tuple_int(strides, 0)
    var s1 = _raw_tuple_int(strides, 1)
    var s2 = _raw_tuple_int(strides, 2)
    var s3 = _raw_tuple_int(strides, 3)
    var ctx = _raw_ctx(device_context_ptr)

    var size = _element_size(dtype)
    if size == 4:
        _permute_copy[DType.uint32](
            out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
        )
    elif size == 2:
        _permute_copy[DType.uint16](
            out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
        )
    elif size == 8:
        _permute_copy[DType.uint64](
            out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
        )
    elif size == 1:
        _permute_copy[DType.uint8](
            out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
        )
    else:
        raise Error("unsupported element size for fast permute")


# ---------------------------------------------------------------------------
# Narrow copy: out is `outer` blocks of `copy_len` contiguous elements taken
# from the source at `outer_index * src_stride + src_offset`. This covers
# split / narrow / step-1 slicing along one dim of a contiguous tensor.
# ---------------------------------------------------------------------------


@always_inline
def _narrow_copy[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    outer: Int,
    src_stride: Int,
    copy_len: Int,
    src_offset: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var total = outer * copy_len

    # Chunk-of-4: one div/mod chain and one (unaligned) vector transfer
    # per 4 output elements; chunks that cross a block boundary fall back
    # to per-element math. Big row slices (e.g. last-token logits) are
    # otherwise division-throughput-bound.
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, total)
    def func4[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value()) * 4
        var o = i // copy_len
        var j = i % copy_len
        if j + 4 <= copy_len and i + 4 <= total:
            out_ptr.store(
                i, in_ptr.load[width=4](o * src_stride + src_offset + j)
            )
        else:
            for u in range(4):
                var iu = i + u
                if iu >= total:
                    return
                var ou = iu // copy_len
                var ju = iu % copy_len
                out_ptr[iu] = in_ptr[ou * src_stride + src_offset + ju]

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = i // copy_len
        var j = i % copy_len
        out_ptr[i] = in_ptr[o * src_stride + src_offset + j]

    if copy_len >= 4:
        _parallel_for[func4]((total + 3) // 4, ctx)
    else:
        _parallel_for[func](total, ctx)


def _narrow_copy_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    outer: PyObjectPtr,
    src_stride: PyObjectPtr,
    copy_len: PyObjectPtr,
    src_offset: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var outer_val = _raw_int(outer)
    var src_stride_val = _raw_int(src_stride)
    var copy_len_val = _raw_int(copy_len)
    var src_offset_val = _raw_int(src_offset)
    var ctx = _raw_ctx(device_context_ptr)

    var size = _element_size(dtype)
    if size == 4:
        _narrow_copy[DType.uint32](
            out_addr,
            in_addr,
            outer_val,
            src_stride_val,
            copy_len_val,
            src_offset_val,
            ctx,
        )
    elif size == 2:
        _narrow_copy[DType.uint16](
            out_addr,
            in_addr,
            outer_val,
            src_stride_val,
            copy_len_val,
            src_offset_val,
            ctx,
        )
    elif size == 8:
        _narrow_copy[DType.uint64](
            out_addr,
            in_addr,
            outer_val,
            src_stride_val,
            copy_len_val,
            src_offset_val,
            ctx,
        )
    elif size == 1:
        _narrow_copy[DType.uint8](
            out_addr,
            in_addr,
            outer_val,
            src_stride_val,
            copy_len_val,
            src_offset_val,
            ctx,
        )
    else:
        raise Error("unsupported element size for fast narrow copy")


# ---------------------------------------------------------------------------
# Narrow copy, destination-strided: the mirror image of _narrow_copy. The
# *source* is fully contiguous (`outer` blocks of `copy_len` elements) and
# lands in the destination at `outer_index * dst_stride + dst_offset`.
# Looping this over the inputs implements concatenation along any dim.
# ---------------------------------------------------------------------------


@always_inline
def _narrow_copy_dst[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    outer: Int,
    dst_stride: Int,
    copy_len: Int,
    dst_offset: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if copy_len % 4 == 0 and dst_stride % 4 == 0 and dst_offset % 4 == 0:
        # Vector fast path: 4 elements per thread with vector-aligned
        # accesses (buffer bases are over-aligned and every index below is
        # a multiple of 4 elements). KV-cache concatenation hits this on
        # each decode step, where the scalar path's per-element div/mod
        # and single-element transactions cost ~2x the bandwidth.
        comptime vec_align = 4 * size_of[dtype]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func4[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value()) * 4
            var o = i // copy_len
            var j = i % copy_len
            var v = in_ptr.load[width=4, alignment=vec_align](i)
            out_ptr.store[width=4, alignment=vec_align](
                o * dst_stride + dst_offset + j, v
            )

        _parallel_for[func4](outer * copy_len // 4, ctx)
        return

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = i // copy_len
        var j = i % copy_len
        out_ptr[o * dst_stride + dst_offset + j] = in_ptr[i]

    _parallel_for[func](outer * copy_len, ctx)


def _narrow_copy_dst_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    outer: PyObjectPtr,
    dst_stride: PyObjectPtr,
    copy_len: PyObjectPtr,
    dst_offset: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(in_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var outer_val = _raw_int(outer)
    var dst_stride_val = _raw_int(dst_stride)
    var copy_len_val = _raw_int(copy_len)
    var dst_offset_val = _raw_int(dst_offset)
    var ctx = _raw_ctx(device_context_ptr)

    var size = _element_size(dtype)
    if size == 4:
        _narrow_copy_dst[DType.uint32](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif size == 2:
        _narrow_copy_dst[DType.uint16](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif size == 8:
        _narrow_copy_dst[DType.uint64](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif size == 1:
        _narrow_copy_dst[DType.uint8](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    else:
        raise Error("unsupported element size for fast narrow copy")


# ---------------------------------------------------------------------------
# Where: out[i] = cond ? a : b. The output is contiguous with dims padded
# to rank 4; cond (bool) and both operands are indexed with their own
# strides (0 on broadcast dims), so scalar operands are 1-element buffers
# with all-zero strides. A pure selection copy — kernels are specialized on
# element byte-size, not dtype.
# ---------------------------------------------------------------------------


@always_inline
def _where_select[
    dtype: DType
](
    out_addr: Int,
    cond_addr: Int,
    a_addr: Int,
    b_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    cs0: Int,
    cs1: Int,
    cs2: Int,
    cs3: Int,
    a_s0: Int,
    a_s1: Int,
    a_s2: Int,
    a_s3: Int,
    b_s0: Int,
    b_s1: Int,
    b_s2: Int,
    b_s3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var cond_ptr = _make_ptr[DType.bool](cond_addr)
    var a_ptr = _make_ptr[dtype](a_addr)
    var b_ptr = _make_ptr[dtype](b_addr)

    # Chunk-of-4 kernel: one thread selects 4 consecutive output elements,
    # so the div/mod coordinate chain (the cost that dominates the scalar
    # version — 6 integer divisions per element) runs once per chunk, and
    # last-dim strides of 1/0 use vector loads / splats.
    var vec_ok = (
        (cs3 == 0 or cs3 == 1)
        and (a_s3 == 0 or a_s3 == 1)
        and (b_s3 == 0 or b_s3 == 1)
        and d3 >= 4
    )

    @always_inline
    @parameter
    @__copy_capture(out_ptr, cond_ptr, a_ptr, b_ptr)
    def func4[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value()) * 4
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        if i3 + 4 <= d3 and i + 4 <= total:
            var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
            var abase = i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3
            var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
            var cond: SIMD[DType.bool, 4]
            if cs3 == 1:
                cond = cond_ptr.load[width=4](cbase)
            else:
                cond = SIMD[DType.bool, 4](cond_ptr[cbase])
            var av: SIMD[dtype, 4]
            if a_s3 == 1:
                av = a_ptr.load[width=4](abase)
            else:
                av = SIMD[dtype, 4](a_ptr[abase])
            var bv: SIMD[dtype, 4]
            if b_s3 == 1:
                bv = b_ptr.load[width=4](bbase)
            else:
                bv = SIMD[dtype, 4](b_ptr[bbase])
            out_ptr.store(i, cond.select(av, bv))
        else:
            # Chunk crosses a row boundary (or the end): per-element math.
            for u in range(4):
                var j = i + u
                if j >= total:
                    return
                var j3 = j % d3
                var jrest = j // d3
                var j2 = jrest % d2
                jrest = jrest // d2
                var j1 = jrest % d1
                var j0 = jrest // d1
                if cond_ptr[j0 * cs0 + j1 * cs1 + j2 * cs2 + j3 * cs3]:
                    out_ptr[j] = a_ptr[
                        j0 * a_s0 + j1 * a_s1 + j2 * a_s2 + j3 * a_s3
                    ]
                else:
                    out_ptr[j] = b_ptr[
                        j0 * b_s0 + j1 * b_s1 + j2 * b_s2 + j3 * b_s3
                    ]

    @always_inline
    @parameter
    @__copy_capture(out_ptr, cond_ptr, a_ptr, b_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        if cond_ptr[i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3]:
            out_ptr[i] = a_ptr[i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3]
        else:
            out_ptr[i] = b_ptr[i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3]

    if vec_ok:
        _parallel_for[func4]((total + 3) // 4, ctx)
    else:
        _parallel_for[func](total, ctx)


def _where_select_go(
    out_buffer: PyObjectPtr,
    cond_buffer: PyObjectPtr,
    a_buffer: PyObjectPtr,
    b_buffer: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, cs0..cs3, as0..as3, bs0..bs3)
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype(out_buffer)
    var out_addr = _raw_addr(out_buffer)
    var cond_addr = _raw_addr(cond_buffer)
    var a_addr = _raw_addr(a_buffer)
    var b_addr = _raw_addr(b_buffer)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var cs0 = _raw_tuple_int(params, 4)
    var cs1 = _raw_tuple_int(params, 5)
    var cs2 = _raw_tuple_int(params, 6)
    var cs3 = _raw_tuple_int(params, 7)
    var a_s0 = _raw_tuple_int(params, 8)
    var a_s1 = _raw_tuple_int(params, 9)
    var a_s2 = _raw_tuple_int(params, 10)
    var a_s3 = _raw_tuple_int(params, 11)
    var b_s0 = _raw_tuple_int(params, 12)
    var b_s1 = _raw_tuple_int(params, 13)
    var b_s2 = _raw_tuple_int(params, 14)
    var b_s3 = _raw_tuple_int(params, 15)
    var total = d0 * d1 * d2 * d3
    var ctx = _raw_ctx(device_context_ptr)

    var size = _element_size(dtype)
    if size == 4:
        _where_select[DType.uint32](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif size == 2:
        _where_select[DType.uint16](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif size == 8:
        _where_select[DType.uint64](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif size == 1:
        _where_select[DType.uint8](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    else:
        raise Error("unsupported element size for fast where")


# ---------------------------------------------------------------------------
# Elementwise dtype cast between contiguous buffers of the same shape.
# ---------------------------------------------------------------------------


@always_inline
def _cast[
    src: DType, dst: DType
](out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dst](out_addr)
    var in_ptr = _make_ptr[src](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr[i]
        comptime if dst == DType.bool:
            out_ptr[i] = Scalar[dst](a != Scalar[src](0))
        else:
            out_ptr[i] = a.cast[dst]()

    _parallel_for[func](size, ctx)


@always_inline
def _cast_to[
    src: DType
](
    dst: DType, out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    if dst == DType.float32:
        _cast[src, DType.float32](out_addr, in_addr, size, ctx)
    elif dst == DType.float16:
        _cast[src, DType.float16](out_addr, in_addr, size, ctx)
    elif dst == DType.bfloat16:
        _cast[src, DType.bfloat16](out_addr, in_addr, size, ctx)
    elif dst == DType.int64:
        _cast[src, DType.int64](out_addr, in_addr, size, ctx)
    elif dst == DType.int32:
        _cast[src, DType.int32](out_addr, in_addr, size, ctx)
    elif dst == DType.uint8:
        _cast[src, DType.uint8](out_addr, in_addr, size, ctx)
    elif dst == DType.bool:
        _cast[src, DType.bool](out_addr, in_addr, size, ctx)
    else:
        raise Error(
            "unsupported destination dtype for fast cast: " + String(dst)
        )


def _cast_go(
    out_buffer: PyObjectPtr,
    in_buffer: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var src = _raw_dtype(in_buffer)
    var dst = _raw_dtype(out_buffer)
    var out_addr = _raw_addr(out_buffer)
    var in_addr = _raw_addr(in_buffer)
    var size = _raw_numel(out_buffer)
    var ctx = _raw_ctx(device_context_ptr)

    if src == DType.float32:
        _cast_to[DType.float32](dst, out_addr, in_addr, size, ctx)
    elif src == DType.float16:
        _cast_to[DType.float16](dst, out_addr, in_addr, size, ctx)
    elif src == DType.bfloat16:
        _cast_to[DType.bfloat16](dst, out_addr, in_addr, size, ctx)
    elif src == DType.int64:
        _cast_to[DType.int64](dst, out_addr, in_addr, size, ctx)
    elif src == DType.int32:
        _cast_to[DType.int32](dst, out_addr, in_addr, size, ctx)
    elif src == DType.uint8:
        _cast_to[DType.uint8](dst, out_addr, in_addr, size, ctx)
    elif src == DType.bool:
        _cast_to[DType.bool](dst, out_addr, in_addr, size, ctx)
    else:
        raise Error("unsupported source dtype for fast cast: " + String(src))


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _permute_copy_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _permute_copy_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _narrow_copy_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _narrow_copy_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except:
        pass
    return _raw_ret_none()


def _narrow_copy_dst_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _narrow_copy_dst_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except:
        pass
    return _raw_ret_none()


def _where_select_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _where_select_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _cast_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _cast_go(args[0], args[1], args[2])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_data_movement_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("data_movement_ops")
        b.def_py_c_function(
            _permute_copy_dispatcher,
            "PermuteCopy",
            docstring=(
                "materialize a permutation of a contiguous tensor (rank <= 4)"
            ),
        )
        b.def_py_c_function(
            _narrow_copy_dispatcher,
            "NarrowCopy",
            docstring=(
                "copy `outer` blocks of `copy_len` elements with a source"
                " stride/offset"
            ),
        )
        b.def_py_c_function(
            _cast_dispatcher,
            "Cast",
            docstring="elementwise dtype cast between contiguous buffers",
        )
        b.def_py_c_function(
            _narrow_copy_dst_dispatcher,
            "NarrowCopyDst",
            docstring=(
                "copy `outer` contiguous blocks of `copy_len` elements to a"
                " destination stride/offset (concatenation)"
            ),
        )
        b.def_py_c_function(
            _where_select_dispatcher,
            "WhereSelect",
            docstring="out = cond ? a : b (broadcast strides, any dtype)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create data_movement_ops python module: {e}")
