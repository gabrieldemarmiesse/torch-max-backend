# ===----------------------------------------------------------------------=== #
# The owning device-memory holder for max_device eager tensors, plus the
# host-transfer and strided-copy primitives that replace the Python
# `max.driver.Buffer`.
#
# Design (see docs/strided_owning_tensors_design.md): a `TensorHolder` is a
# *pure ownership token* — it owns one `DeviceBuffer[DType.uint8]` (byte-typed
# = dtype-erased) allocated on MAX's stream via `enqueue_create_buffer`. All
# layout metadata (shape / strides / storage_offset / dtype) lives on the
# Python `TorchMojoTensor` wrapper; views share the *same* holder object and
# CPython's refcount keeps the allocation alive until the last view dies, at
# which point the holder's destructor releases the AsyncRT buffer — a
# stream-ordered (fire-and-forget) free that is safe because alloc, kernels
# and free all ride the device's one default stream.
#
# Kernels never see the holder: Python passes raw data pointers (with the
# storage offset already applied) as ints. The holder/spec struct definitions
# live in `op_utils` (single shared source, compiled into every kernel module
# that registers them); this module registers them for Python and owns
# `make_spec`, the one Python-facing spec constructor.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.memory import OpaquePointer
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.utils.coord import Coord

from std.algorithm.functional import elementwise
from std.sys.info import has_accelerator, size_of
from std.utils import IndexList

from op_utils import (
    MAX_RANK,
    TensorHolder,
    TensorSpec,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _row_major8,
    _spec_ptr,
    _spec_result,
    _spec_unsupported,
)

# Every dtype a max_device tensor can hold (read_scalar / StridedFill
# dispatch over this at compile time).
comptime ALL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.bool,
]


@always_inline
def _ctx_from(ctx_ptr: PythonObject) raises -> DeviceContext:
    from std.memory import OpaquePointer

    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=Int(py=ctx_ptr))
    )


@always_inline
def _u8_ptr(
    addr: Int,
) -> UnsafePointer[Scalar[DType.uint8], MutUntrackedOrigin]:
    return _make_ptr[DType.uint8](addr)


@always_inline
def _wrap_raw(
    ctx: DeviceContext, addr: Int, nbytes: Int
) -> DeviceBuffer[DType.uint8]:
    """A non-owning DeviceBuffer view over raw device memory, for copies."""
    return DeviceBuffer[DType.uint8](ctx, _u8_ptr(addr), nbytes, owning=False)


# ---------------------------------------------------------------------------
# Allocation + host transfers. All return/accept raw addresses as Python
# ints; the holder is only ever returned as the ownership token.
# ---------------------------------------------------------------------------


def alloc(ctx_ptr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    """Allocate owning device memory. Returns (holder, data_ptr)."""
    var ctx = _ctx_from(ctx_ptr)
    var n = Int(py=nbytes)
    # Zero-sized tensors still get a real 1-byte allocation so every tensor
    # carries a valid pointer (empty driver.Buffers had sentinel pointers
    # that didn't survive round-trips).
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(n, 1))
    var addr = Int(buf.unsafe_ptr())
    var holder = PythonObject(alloc=TensorHolder(buf=buf^, nbytes=n))
    return Python.tuple(holder^, PythonObject(addr))


def alloc_from_host(
    ctx_ptr: PythonObject, host_ptr: PythonObject, nbytes: PythonObject
) raises -> PythonObject:
    """Allocate + H2D copy from raw host memory. Returns (holder, data_ptr).

    Synchronizes before returning: the host memory (typically a CPU torch
    tensor's storage) is only guaranteed alive for the duration of the call.
    """
    var ctx = _ctx_from(ctx_ptr)
    var n = Int(py=nbytes)
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(n, 1))
    if n > 0:
        buf.enqueue_copy_from(_u8_ptr(Int(py=host_ptr)))
        ctx.synchronize()
    var addr = Int(buf.unsafe_ptr())
    var holder = PythonObject(alloc=TensorHolder(buf=buf^, nbytes=n))
    return Python.tuple(holder^, PythonObject(addr))


def copy_from_host(
    ctx_ptr: PythonObject,
    dev_ptr: PythonObject,
    host_ptr: PythonObject,
    nbytes: PythonObject,
) raises:
    """H2D copy into existing device memory. Synchronizes (see above)."""
    var n = Int(py=nbytes)
    if n == 0:
        return
    var ctx = _ctx_from(ctx_ptr)
    var dst = _wrap_raw(ctx, Int(py=dev_ptr), n)
    dst.enqueue_copy_from(_u8_ptr(Int(py=host_ptr)))
    ctx.synchronize()


def copy_to_host(
    ctx_ptr: PythonObject,
    dev_ptr: PythonObject,
    host_ptr: PythonObject,
    nbytes: PythonObject,
) raises:
    """D2H copy into raw host memory. Synchronizes before returning."""
    var n = Int(py=nbytes)
    if n == 0:
        return
    var ctx = _ctx_from(ctx_ptr)
    var src = _wrap_raw(ctx, Int(py=dev_ptr), n)
    src.enqueue_copy_to(_u8_ptr(Int(py=host_ptr)))
    ctx.synchronize()


def copy_d2d(
    ctx_ptr: PythonObject,
    dst_ptr: PythonObject,
    src_ptr: PythonObject,
    nbytes: PythonObject,
) raises:
    """Device-to-device copy on one context.

    Stream-ordered with no sync on GPU. The CPU device runs copies on a
    worker pool that is NOT ordered with kernel execution, so there the
    copy must complete before returning or a later kernel writing the same
    buffer can be overwritten by it (seen as select_scatter flakes under
    parallel test load)."""
    var n = Int(py=nbytes)
    if n == 0:
        return
    var ctx = _ctx_from(ctx_ptr)
    var dst = _wrap_raw(ctx, Int(py=dst_ptr), n)
    var src = _wrap_raw(ctx, Int(py=src_ptr), n)
    dst.enqueue_copy_from(src)
    if ctx.api() == "cpu":
        ctx.synchronize()


def synchronize(ctx_ptr: PythonObject) raises:
    _ctx_from(ctx_ptr).synchronize()


def read_scalar(
    ctx_ptr: PythonObject, dev_ptr: PythonObject, dtype_value: PythonObject
) raises -> PythonObject:
    """Read one element at dev_ptr as a Python bool/int/float (syncs)."""
    var ctx = _ctx_from(ctx_ptr)
    var dtype = DType._from_ui8(UInt8(Int(py=dtype_value))._mlir_value)
    var staging = InlineArray[UInt8, 16](fill=0)

    comptime for dt in ALL_DTYPES:
        if dtype == dt:
            var src = _wrap_raw(ctx, Int(py=dev_ptr), size_of[dt]())
            src.enqueue_copy_to(staging.unsafe_ptr())
            ctx.synchronize()
            var val = staging.unsafe_ptr().bitcast[Scalar[dt]]()[0]
            comptime if dt == DType.bool:
                return PythonObject(Bool(val))
            elif dt.is_floating_point():
                return PythonObject(Float64(val.cast[DType.float64]()))
            else:
                return PythonObject(Int(val))
    raise Error("read_scalar: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# CopyStrided: dst[coords] = src[coords] over an arbitrary-rank-<=8 index
# space, with independent element strides on both sides (0-stride broadcast
# reads included). This is the shared materialize primitive: .contiguous(),
# copy_ into strided destinations, expand materialization — everything that
# moves elements between two layouts of the same dtype.
#
# Layout-only, so it dispatches on element *size*, not dtype.
#
# Raw METH_FASTCALL args:
#   (dst_ptr, src_ptr, shape8, dst_strides8, src_strides8, itemsize, ctx_ptr)
# shape8/dst_strides8/src_strides8 are int tuples padded to MAX_RANK
# leading entries (size 1 / stride 0). Pointers are element-aligned ints
# with any storage offset already applied.
# ---------------------------------------------------------------------------


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


@always_inline
def _copy_strided[
    dtype: DType
](
    dst_addr: Int,
    src_addr: Int,
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    src_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    var dst_ptr = _make_ptr[dtype](dst_addr)
    var src_ptr = _make_ptr[dtype](src_addr)
    var total = 1
    for i in range(MAX_RANK):
        total *= shape[i]
    if total == 0:
        return

    @always_inline
    @parameter
    @__copy_capture(dst_ptr, src_ptr, shape, dst_strides, src_strides)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var rest = Int(idx[0].value())
        var dst_off = 0
        var src_off = 0

        comptime for d in range(MAX_RANK - 1, 0, -1):
            var coord = rest % shape[d]
            rest = rest // shape[d]
            dst_off += coord * dst_strides[d]
            src_off += coord * src_strides[d]
        dst_off += rest * dst_strides[0]
        src_off += rest * src_strides[0]
        dst_ptr[dst_off] = src_ptr[src_off]

    _parallel_for[func](total, ctx)


def _copy_strided_go(
    dst_ptr: PyObjectPtr,
    src_ptr: PyObjectPtr,
    shape_t: PyObjectPtr,
    dst_strides_t: PyObjectPtr,
    src_strides_t: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var src_addr = _raw_int(src_ptr)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    var src_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
        src_strides[i] = _raw_tuple_int(src_strides_t, i)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    if itemsize == 4:
        _copy_strided[DType.uint32](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 2:
        _copy_strided[DType.uint16](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 8:
        _copy_strided[DType.uint64](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 1:
        _copy_strided[DType.uint8](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    else:
        raise Error("CopyStrided: unsupported element size ", itemsize)


def _copy_strided_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _copy_strided_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# StridedFill: dst[coords] = value over a strided rank-<=8 destination.
# Needs the real dtype (the Float64 value is cast once). Raw args:
#   (dst_ptr, value_f64, shape8, dst_strides8, dtype_value, ctx_ptr)
# ---------------------------------------------------------------------------


@always_inline
def _strided_fill[
    dtype: DType
](
    dst_addr: Int,
    value: Float64,
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    var dst_ptr = _make_ptr[dtype](dst_addr)
    var scalar = value.cast[dtype]()
    var total = 1
    for i in range(MAX_RANK):
        total *= shape[i]
    if total == 0:
        return

    @always_inline
    @parameter
    @__copy_capture(dst_ptr, scalar, shape, dst_strides)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var rest = Int(idx[0].value())
        var dst_off = 0

        comptime for d in range(MAX_RANK - 1, 0, -1):
            var coord = rest % shape[d]
            rest = rest // shape[d]
            dst_off += coord * dst_strides[d]
        dst_off += rest * dst_strides[0]
        dst_ptr[dst_off] = scalar

    _parallel_for[func](total, ctx)


def _strided_fill_go(
    dst_ptr: PyObjectPtr,
    value_o: PyObjectPtr,
    shape_t: PyObjectPtr,
    dst_strides_t: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var value = _raw_f64(value_o)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
    var dtype = DType._from_ui8(UInt8(_raw_int(dtype_o))._mlir_value)
    var ctx = _raw_ctx(ctx_ptr)

    # bool fill must store exactly 0/1.
    if dtype == DType.bool:
        value = Float64(1) if value != 0 else Float64(0)

    var handled = False
    comptime for dt in ALL_DTYPES:
        if dtype == dt:
            _strided_fill[dt](dst_addr, value, shape, dst_strides, ctx)
            handled = True
    if not handled:
        raise Error("StridedFill: unsupported dtype ", dtype)


def _strided_fill_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _strided_fill_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


# ===========================================================================
# TensorSpec constructor + the POC spec ops (add / mul / relu / batch_norm).
# The shared infrastructure (TensorSpec/TensorHolder structs, _spec_ptr,
# _spec_result, _spec_unsupported) lives in `op_utils`; per-family spec ops
# migrate next to their kernels (see docs/tensor_spec_design.md §3).
# ===========================================================================


def _make_spec_go(
    ptr_o: PyObjectPtr,
    rank_o: PyObjectPtr,
    shape_t: PyObjectPtr,
    strides_t: PyObjectPtr,
    offset_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    numel_o: PyObjectPtr,
    contig_o: PyObjectPtr,
    ctx_o: PyObjectPtr,
) raises -> PyObjectPtr:
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        strides[i] = _raw_tuple_int(strides_t, i)
    var obj = PythonObject(
        alloc=TensorSpec(
            ptr=_raw_int(ptr_o),
            rank=_raw_int(rank_o),
            shape=shape,
            strides=strides,
            offset=_raw_int(offset_o),
            dtype=_raw_dtype_int(dtype_o),
            itemsize=_raw_int(itemsize_o),
            numel=_raw_int(numel_o),
            contig=_raw_int(contig_o) != 0,
            ctx_ptr=_raw_int(ctx_o),
        )
    )
    return obj^.steal_data()


def _make_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _make_spec_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
        )
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Binary broadcast spec ops (add / mul). One entry replaces the Python-side
# same-shape gate, `_bcast_meta`, output alloc and kernel launch: broadcast
# degenerates to same-shape naturally, and leading-padded rank-8 layouts
# align at the trailing edge with no rank bookkeeping at all.
# ---------------------------------------------------------------------------

comptime SOP_ADD = 0
comptime SOP_MUL = 1

comptime SPEC_BIN_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]


@always_inline
def _bin_bcast_spec[
    dtype: DType, op_code: Int
](
    out_addr: Int,
    l_addr: Int,
    r_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    ls0: Int,
    ls1: Int,
    ls2: Int,
    ls3: Int,
    rs0: Int,
    rs1: Int,
    rs2: Int,
    rs3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var l_ptr = _make_ptr[dtype](l_addr)
    var r_ptr = _make_ptr[dtype](r_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, l_ptr, r_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var a = l_ptr[i0 * ls0 + i1 * ls1 + i2 * ls2 + i3 * ls3]
        var b = r_ptr[i0 * rs0 + i1 * rs1 + i2 * rs2 + i3 * rs3]
        comptime if op_code == SOP_ADD:
            out_ptr[i] = a + b
        comptime if op_code == SOP_MUL:
            out_ptr[i] = a * b

    _parallel_for[func](total, ctx)


def _binary_spec_go[
    op_code: Int
](a_o: PyObjectPtr, b_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    ref b = _spec_ptr(b_o)[]

    if a.dtype != b.dtype:
        raise Error("mojo spec binary: operand dtypes differ")
    if a.ctx_ptr != b.ctx_ptr:
        raise Error("mojo spec binary: operands on different devices")
    if a.rank > 4 or b.rank > 4:
        raise Error("mojo spec binary: rank > 4")
    var supported = False
    comptime for dt in SPEC_BIN_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec binary: unsupported dtype ", a.dtype)

    # Broadcast over the trailing 4 slots; leading padding aligns ranks.
    var d = IndexList[4](1)
    var ls = IndexList[4](0)
    var rs = IndexList[4](0)
    for k in range(4):
        var i = MAX_RANK - 4 + k
        var sa = a.shape[i]
        var sb = b.shape[i]
        var s: Int
        if sa == sb:
            s = sa
        elif sa == 1:
            s = sb
        elif sb == 1:
            s = sa
        else:
            raise Error("mojo spec binary: shapes do not broadcast")
        d[k] = s
        ls[k] = a.strides[i] if sa != 1 else 0
        rs[k] = b.strides[i] if sb != 1 else 0

    var out_rank = max(a.rank, b.rank)
    var numel = d[0] * d[1] * d[2] * d[3]
    var nbytes = numel * a.itemsize
    var ctx = a.ctx()
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())

    if numel > 0:
        comptime for dt in SPEC_BIN_DTYPES:
            if a.dtype == dt:
                _bin_bcast_spec[dt, op_code](
                    addr,
                    a.ptr,
                    b.ptr,
                    d[1],
                    d[2],
                    d[3],
                    ls[0],
                    ls[1],
                    ls[2],
                    ls[3],
                    rs[0],
                    rs[1],
                    rs[2],
                    rs[3],
                    numel,
                    ctx,
                )

    var oshape = IndexList[MAX_RANK](1)
    for k in range(4):
        oshape[MAX_RANK - 4 + k] = d[k]
    return _spec_result(
        buf^,
        addr,
        nbytes,
        out_rank,
        oshape,
        a.dtype,
        a.itemsize,
        numel,
        a.ctx_ptr,
    )


def _add_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _binary_spec_go[SOP_ADD](args[0], args[1])
    except e:
        return _spec_unsupported(e)


def _mul_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _binary_spec_go[SOP_MUL](args[0], args[1])
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Relu spec op: contiguous unary, checks + alloc + launch in one call.
# ---------------------------------------------------------------------------


@always_inline
def _relu_contig[
    dtype: DType
](out_addr: Int, in_addr: Int, total: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.load[width=width](i)
        out_ptr.store[width=width](i, max(a, SIMD[dtype, width](0)))

    _parallel_for[func](total, ctx)


def _relu_spec_go(a_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]

    if not a.contig:
        raise Error("mojo spec relu: input not contiguous")
    var supported = False
    comptime for dt in SPEC_BIN_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec relu: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        comptime for dt in SPEC_BIN_DTYPES:
            if a.dtype == dt:
                _relu_contig[dt](addr, a.ptr, a.numel, ctx)
    return _spec_result(
        buf^,
        addr,
        nbytes,
        a.rank,
        a.shape,
        a.dtype,
        a.itemsize,
        a.numel,
        a.ctx_ptr,
    )


def _relu_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _relu_spec_go(args[0])
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Batch norm (inference) spec op: geometry (channels/inner) derived from the
# input spec instead of being computed in Python and smuggled in a tuple.
# ---------------------------------------------------------------------------

comptime SPEC_FLOAT_DTYPES = [DType.float32, DType.float16, DType.bfloat16]


@always_inline
def _batch_norm_spec_kernel[
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


def _batch_norm_spec_go(
    in_o: PyObjectPtr,
    mean_o: PyObjectPtr,
    var_o: PyObjectPtr,
    gamma_o: PyObjectPtr,
    beta_o: PyObjectPtr,
    eps_o: PyObjectPtr,
) raises -> PyObjectPtr:
    ref inp = _spec_ptr(in_o)[]
    ref meanp = _spec_ptr(mean_o)[]
    ref varp = _spec_ptr(var_o)[]
    ref gammap = _spec_ptr(gamma_o)[]
    ref betap = _spec_ptr(beta_o)[]
    var eps = Float32(_raw_f64(eps_o))

    if inp.rank < 2:
        raise Error("mojo spec batch_norm: input rank must be >= 2")
    if inp.numel == 0:
        raise Error("mojo spec batch_norm: empty input")
    if not (
        inp.contig
        and meanp.contig
        and varp.contig
        and gammap.contig
        and betap.contig
    ):
        raise Error("mojo spec batch_norm: all inputs must be contiguous")
    if (
        meanp.dtype != inp.dtype
        or varp.dtype != inp.dtype
        or gammap.dtype != inp.dtype
        or betap.dtype != inp.dtype
    ):
        raise Error("mojo spec batch_norm: stat/affine dtypes must match input")
    var supported = False
    comptime for dt in SPEC_FLOAT_DTYPES:
        if inp.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec batch_norm: unsupported dtype ", inp.dtype)

    var channels = inp.dim(1)
    var inner = 1
    for i in range(MAX_RANK - inp.rank + 2, MAX_RANK):
        inner *= inp.shape[i]

    var ctx = inp.ctx()
    var nbytes = inp.numel * inp.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    comptime for dt in SPEC_FLOAT_DTYPES:
        if inp.dtype == dt:
            _batch_norm_spec_kernel[dt](
                addr,
                inp.ptr,
                meanp.ptr,
                varp.ptr,
                gammap.ptr,
                betap.ptr,
                eps,
                channels,
                inner,
                inp.numel,
                ctx,
            )
    return _spec_result(
        buf^,
        addr,
        nbytes,
        inp.rank,
        inp.shape,
        inp.dtype,
        inp.itemsize,
        inp.numel,
        inp.ctx_ptr,
    )


def _batch_norm_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _batch_norm_spec_go(
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_tensor_holder() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("tensor_holder")
        _ = (
            m.add_type[TensorHolder]("TensorHolder")
            .def_method[TensorHolder.data_ptr]("data_ptr")
            .def_method[TensorHolder.get_nbytes]("get_nbytes")
        )
        _ = m.add_type[TensorSpec]("TensorSpec")
        m.def_py_c_function(
            _make_spec_dispatcher,
            "make_spec",
            docstring=(
                "(ptr, rank, shape8, strides8, offset, dtype, itemsize,"
                " numel, contig, ctx_ptr) -> TensorSpec"
            ),
        )
        m.def_py_c_function(
            _add_spec_dispatcher,
            "AddSpec",
            docstring=(
                "(a_spec, b_spec) -> (holder, spec, shape, ptr); broadcast"
                " add with checks/alloc/launch in Mojo"
            ),
        )
        m.def_py_c_function(
            _mul_spec_dispatcher,
            "MulSpec",
            docstring=(
                "(a_spec, b_spec) -> (holder, spec, shape, ptr); broadcast"
                " mul with checks/alloc/launch in Mojo"
            ),
        )
        m.def_py_c_function(
            _relu_spec_dispatcher,
            "ReluSpec",
            docstring=(
                "(a_spec) -> (holder, spec, shape, ptr); contiguous relu"
                " with checks/alloc/launch in Mojo"
            ),
        )
        m.def_py_c_function(
            _batch_norm_spec_dispatcher,
            "BatchNormSpec",
            docstring=(
                "(in, mean, var, gamma, beta specs, eps) -> (holder, spec,"
                " shape, ptr); inference batch norm, geometry from specs"
            ),
        )
        m.def_function[alloc]("alloc")
        m.def_function[alloc_from_host]("alloc_from_host")
        m.def_function[copy_from_host]("copy_from_host")
        m.def_function[copy_to_host]("copy_to_host")
        m.def_function[copy_d2d]("copy_d2d")
        m.def_function[synchronize]("synchronize")
        m.def_function[read_scalar]("read_scalar")
        m.def_py_c_function(
            _copy_strided_dispatcher,
            "CopyStrided",
            docstring=(
                "dst[coords] = src[coords] over rank-8-padded strided"
                " layouts (element-size dispatch)"
            ),
        )
        m.def_py_c_function(
            _strided_fill_dispatcher,
            "StridedFill",
            docstring=(
                "dst[coords] = value over a rank-8-padded strided layout"
                " (dtype dispatch)"
            ),
        )
        return m.finalize()
    except e:
        abort(t"failed to create tensor_holder python module: {e}")
