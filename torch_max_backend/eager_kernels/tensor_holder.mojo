# ===----------------------------------------------------------------------=== #
# The owning device-memory holder for max_device eager tensors, plus the
# host-transfer and strided-copy primitives that replace the Python
# `max.driver.Buffer`.
#
# Design (see docs/strided_owning_tensors_design.md): a `TensorHolder` is a
# *pure ownership token* — it owns one `DeviceBuffer[DType.uint8]` (byte-typed
# = dtype-erased) allocated on MAX's stream via `enqueue_create_buffer`. All
# layout metadata (shape / strides / storage_offset / dtype) lives on the
# Python `TorchMaxTensor` wrapper; views share the *same* holder object and
# CPython's refcount keeps the allocation alive until the last view dies, at
# which point the holder's destructor releases the AsyncRT buffer — a
# stream-ordered (fire-and-forget) free that is safe because alloc, kernels
# and free all ride the device's one default stream.
#
# Kernels never see the holder: Python passes raw data pointers (with the
# storage offset already applied) as ints. This module is therefore the only
# place the holder type is registered (single-registration rule).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext, DeviceBuffer
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.utils.coord import Coord

from std.algorithm.functional import elementwise
from std.sys.info import has_accelerator, size_of
from std.utils import IndexList

from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
)

# Strided kernels always work on shapes/strides padded to this rank by the
# Python side (leading dims of size 1 / stride 0).
comptime MAX_RANK = 8

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


struct TensorHolder(Movable, Writable):
    """Owns one device allocation. Nothing else.

    `buf`'s destructor (run by this struct's destructor when the CPython
    refcount hits 0) calls `AsyncRT_DeviceBuffer_release`, which enqueues
    the stream-ordered free.
    """

    var buf: DeviceBuffer[DType.uint8]
    var nbytes: Int

    def __init__(out self, var buf: DeviceBuffer[DType.uint8], nbytes: Int):
        self.buf = buf^
        self.nbytes = nbytes

    def write_to(self, mut writer: Some[Writer]):
        # Writable is mandatory for types exposed via add_type (tp_repr).
        writer.write(
            "TensorHolder(ptr=",
            Int(self.buf.unsafe_ptr()),
            ", nbytes=",
            self.nbytes,
            ")",
        )

    @staticmethod
    def data_ptr(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        return PythonObject(Int(self_ptr[].buf.unsafe_ptr()))

    @staticmethod
    def get_nbytes(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        return PythonObject(self_ptr[].nbytes)


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
    """Device-to-device copy on one context. Stream-ordered, no sync."""
    var n = Int(py=nbytes)
    if n == 0:
        return
    var ctx = _ctx_from(ctx_ptr)
    var dst = _wrap_raw(ctx, Int(py=dst_ptr), n)
    var src = _wrap_raw(ctx, Int(py=src_ptr), n)
    dst.enqueue_copy_from(src)


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
