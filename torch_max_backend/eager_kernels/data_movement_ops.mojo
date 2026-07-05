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
from std.sys.info import has_accelerator
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

from op_utils import _get_ctx, _get_dtype, _make_ptr


@always_inline
def _element_size(dtype: DType) raises -> Int:
    """Element size in bytes; copies dispatch on size, not dtype."""
    if (
        dtype == DType.float32
        or dtype == DType.int32
        or dtype == DType.uint32
    ):
        return 4
    if (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.int16
        or dtype == DType.uint16
    ):
        return 2
    if (
        dtype == DType.float64
        or dtype == DType.int64
        or dtype == DType.uint64
    ):
        return 8
    if dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 1
    raise Error("unsupported dtype for fast copy: " + String(dtype))


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1] (Coord) capturing [_] -> None
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


def _permute_copy_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    dims: PythonObject,
    strides: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var d0 = Int(py=dims[0])
    var d1 = Int(py=dims[1])
    var d2 = Int(py=dims[2])
    var d3 = Int(py=dims[3])
    var s0 = Int(py=strides[0])
    var s1 = Int(py=strides[1])
    var s2 = Int(py=strides[2])
    var s3 = Int(py=strides[3])
    var ctx = _get_ctx(device_context_ptr)

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

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = i // copy_len
        var j = i % copy_len
        out_ptr[i] = in_ptr[o * src_stride + src_offset + j]

    _parallel_for[func](outer * copy_len, ctx)


def _narrow_copy_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    outer: PythonObject,
    src_stride: PythonObject,
    copy_len: PythonObject,
    src_offset: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var outer_val = Int(py=outer)
    var src_stride_val = Int(py=src_stride)
    var copy_len_val = Int(py=copy_len)
    var src_offset_val = Int(py=src_offset)
    var ctx = _get_ctx(device_context_ptr)

    var size = _element_size(dtype)
    if size == 4:
        _narrow_copy[DType.uint32](
            out_addr, in_addr, outer_val, src_stride_val, copy_len_val,
            src_offset_val, ctx,
        )
    elif size == 2:
        _narrow_copy[DType.uint16](
            out_addr, in_addr, outer_val, src_stride_val, copy_len_val,
            src_offset_val, ctx,
        )
    elif size == 8:
        _narrow_copy[DType.uint64](
            out_addr, in_addr, outer_val, src_stride_val, copy_len_val,
            src_offset_val, ctx,
        )
    elif size == 1:
        _narrow_copy[DType.uint8](
            out_addr, in_addr, outer_val, src_stride_val, copy_len_val,
            src_offset_val, ctx,
        )
    else:
        raise Error("unsupported element size for fast narrow copy")


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
        raise Error("unsupported destination dtype for fast cast: " + String(dst))


def _cast_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var src = _get_dtype(in_buffer)
    var dst = _get_dtype(out_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var size = Int(py=out_buffer.num_elements)
    var ctx = _get_ctx(device_context_ptr)

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
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_data_movement_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("data_movement_ops")
        b.def_function[_permute_copy_dispatcher](
            "PermuteCopy",
            docstring="materialize a permutation of a contiguous tensor (rank <= 4)",
        )
        b.def_function[_narrow_copy_dispatcher](
            "NarrowCopy",
            docstring="copy `outer` blocks of `copy_len` elements with a source stride/offset",
        )
        b.def_function[_cast_dispatcher](
            "Cast", docstring="elementwise dtype cast between contiguous buffers"
        )
        return b.finalize()
    except e:
        abort(t"failed to create data_movement_ops python module: {e}")
