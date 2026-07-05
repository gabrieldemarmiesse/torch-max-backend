# ===----------------------------------------------------------------------=== #
# Fast eager-mode conv2d for max_device, backed by the MAX kernel library
# (`nn.conv.conv_gpu`). With `filter_is_fcrs=True` the filter is taken in
# PyTorch's native (K, C, R, S) layout and, on NVIDIA GPUs where no
# specialized kernel applies, the call routes to cuDNN — so torch conv
# weights can be passed through unchanged. Input and output are NHWC; the
# Python side permutes activations around the call.
#
# GPU only: the Python side falls back to the graph path on CPU devices.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord as StdCoord
from std.utils.index import IndexList

from std.algorithm.functional import elementwise

from layout import TileTensor, row_major

from nn.conv.conv import conv_gpu

from op_utils import _get_ctx, _get_dtype, _make_ptr


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1] (StdCoord) capturing [_] -> None
](count: Int, ctx: DeviceContext) raises:
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](StdCoord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](StdCoord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _conv2d[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    filt_addr: Int,
    n: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    kh: Int,
    kw: Int,
    out_h: Int,
    out_w: Int,
    stride_h: Int,
    stride_w: Int,
    dil_h: Int,
    dil_w: Int,
    pad_h: Int,
    pad_w: Int,
    groups: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        var input = TileTensor(
            _make_ptr[dtype](in_addr), row_major(n, in_h, in_w, in_c)
        )
        # PyTorch layout: (out_c, in_c / groups, kh, kw) == FCRS.
        var filt = TileTensor(
            _make_ptr[dtype](filt_addr),
            row_major(out_c, in_c // groups, kh, kw),
        )
        var output = TileTensor(
            _make_ptr[dtype](out_addr), row_major(n, out_h, out_w, out_c)
        )
        conv_gpu[dtype, dtype, dtype, filter_is_fcrs=True](
            input,
            filt,
            output,
            IndexList[2](stride_h, stride_w),
            IndexList[2](dil_h, dil_w),
            IndexList[4](pad_h, pad_h, pad_w, pad_w),
            groups,
            ctx,
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def _conv2d_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    filt_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var filt_addr = Int(py=filt_buffer._data_ptr())
    var n = Int(py=params[0])
    var in_h = Int(py=params[1])
    var in_w = Int(py=params[2])
    var in_c = Int(py=params[3])
    var out_c = Int(py=params[4])
    var kh = Int(py=params[5])
    var kw = Int(py=params[6])
    var out_h = Int(py=params[7])
    var out_w = Int(py=params[8])
    var stride_h = Int(py=params[9])
    var stride_w = Int(py=params[10])
    var dil_h = Int(py=params[11])
    var dil_w = Int(py=params[12])
    var pad_h = Int(py=params[13])
    var pad_w = Int(py=params[14])
    var groups = Int(py=params[15])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _conv2d[DType.float32](
            out_addr, in_addr, filt_addr, n, in_h, in_w, in_c, out_c, kh, kw,
            out_h, out_w, stride_h, stride_w, dil_h, dil_w, pad_h, pad_w,
            groups, ctx,
        )
    elif dtype == DType.float16:
        _conv2d[DType.float16](
            out_addr, in_addr, filt_addr, n, in_h, in_w, in_c, out_c, kh, kw,
            out_h, out_w, stride_h, stride_w, dil_h, dil_w, pad_h, pad_w,
            groups, ctx,
        )
    elif dtype == DType.bfloat16:
        _conv2d[DType.bfloat16](
            out_addr, in_addr, filt_addr, n, in_h, in_w, in_c, out_c, kh, kw,
            out_h, out_w, stride_h, stride_w, dil_h, dil_w, pad_h, pad_w,
            groups, ctx,
        )
    else:
        raise Error("unsupported dtype for fast conv2d: " + String(dtype))


# ---------------------------------------------------------------------------
# im2col for a single NCHW sample: builds the (C*KH*KW, OH*OW) patch matrix
# so that conv = weight.view(K, C*KH*KW) @ col. Row order matches the
# reduction order of torch's (K, C, KH, KW) filter, so the weight can be
# used as-is (zero copy) and the matmul output is already NCHW.
# ---------------------------------------------------------------------------


@always_inline
def _im2col[
    dtype: DType
](
    out_addr: Int,
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
    channels: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        var cols = out_h * out_w
        var r = i // cols
        var j = i % cols
        var fw = r % kw
        var fh = (r // kw) % kh
        var c = r // (kw * kh)
        var oh = j // out_w
        var ow = j % out_w
        var ih = oh * stride_h - pad_h + fh * dil_h
        var iw = ow * stride_w - pad_w + fw * dil_w
        if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
            out_ptr[i] = Scalar[dtype](0)
        else:
            out_ptr[i] = in_ptr[(c * in_h + ih) * in_w + iw]

    _parallel_for[func](channels * kh * kw * out_h * out_w, ctx)


def _im2col_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var in_h = Int(py=params[0])
    var in_w = Int(py=params[1])
    var out_h = Int(py=params[2])
    var out_w = Int(py=params[3])
    var kh = Int(py=params[4])
    var kw = Int(py=params[5])
    var stride_h = Int(py=params[6])
    var stride_w = Int(py=params[7])
    var pad_h = Int(py=params[8])
    var pad_w = Int(py=params[9])
    var dil_h = Int(py=params[10])
    var dil_w = Int(py=params[11])
    var channels = Int(py=params[12])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _im2col[DType.float32](
            out_addr, in_addr, in_h, in_w, out_h, out_w, kh, kw, stride_h,
            stride_w, pad_h, pad_w, dil_h, dil_w, channels, ctx,
        )
    elif dtype == DType.float16:
        _im2col[DType.float16](
            out_addr, in_addr, in_h, in_w, out_h, out_w, kh, kw, stride_h,
            stride_w, pad_h, pad_w, dil_h, dil_w, channels, ctx,
        )
    elif dtype == DType.bfloat16:
        _im2col[DType.bfloat16](
            out_addr, in_addr, in_h, in_w, out_h, out_w, kh, kw, stride_h,
            stride_w, pad_h, pad_w, dil_h, dil_w, channels, ctx,
        )
    else:
        raise Error("unsupported dtype for fast im2col: " + String(dtype))


# ---------------------------------------------------------------------------
# In-place per-channel bias add on a (channels, plane) matrix:
# out[i] += bias[i // plane].
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_chan[
    dtype: DType
](out_addr: Int, bias_addr: Int, total: Int, plane: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[i] = out_ptr[i] + bias_ptr[i // plane]

    _parallel_for[func](total, ctx)


def _bias_add_chan_dispatcher(
    out_buffer: PythonObject,
    bias_buffer: PythonObject,
    plane: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(out_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var bias_addr = Int(py=bias_buffer._data_ptr())
    var total = Int(py=out_buffer.num_elements)
    var plane_val = Int(py=plane)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _bias_add_chan[DType.float32](out_addr, bias_addr, total, plane_val, ctx)
    elif dtype == DType.float16:
        _bias_add_chan[DType.float16](out_addr, bias_addr, total, plane_val, ctx)
    elif dtype == DType.bfloat16:
        _bias_add_chan[DType.bfloat16](out_addr, bias_addr, total, plane_val, ctx)
    else:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_conv_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("conv_ops")
        b.def_function[_conv2d_dispatcher](
            "Conv2d",
            docstring="conv2d, NHWC activations + PyTorch (K,C,R,S) filter, GPU only",
        )
        b.def_function[_im2col_dispatcher](
            "Im2col",
            docstring="single-sample NCHW im2col -> (C*KH*KW, OH*OW) patch matrix",
        )
        b.def_function[_bias_add_chan_dispatcher](
            "BiasAddChan",
            docstring="in-place out[i] += bias[i // plane] on a (channels, plane) matrix",
        )
        return b.finalize()
    except e:
        abort(t"failed to create conv_ops python module: {e}")
