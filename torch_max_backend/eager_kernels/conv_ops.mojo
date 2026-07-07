# ===----------------------------------------------------------------------=== #
# Fast eager-mode conv2d support kernels for max_device — pure Mojo, no
# cuDNN. Convolution is lowered to (batched) im2col + the pure-Mojo GEMM in
# `matmul_ops`: the torch (K, C, R, S) weight is used as-is (the im2col row
# order matches its reduction order) and the matmul output is already NCHW.
#
# GPU only: the Python side falls back to the graph path on CPU devices.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord as StdCoord

from std.algorithm.functional import elementwise

from op_utils import FLOAT_DTYPES, _get_ctx, _get_dtype, _make_ptr


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1](StdCoord) capturing[_] -> None
](count: Int, ctx: DeviceContext) raises:
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](StdCoord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](StdCoord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Batched im2col for NCHW input: builds the (N, C*KH*KW, OH*OW) patch
# matrix so that conv = weight.view(K, C*KH*KW) @ col[s]. Row order matches
# the reduction order of torch's (K, C, KH, KW) filter, so the weight can
# be used as-is (zero copy) and the matmul output is already NCHW. Rows are
# channel-major, so grouped convolution can slice the row range of each
# group with a plain element offset.
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
    batch: Int,
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
        var crs = channels * kh * kw
        var s = i // (crs * cols)
        var r = (i // cols) % crs
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
            out_ptr[i] = in_ptr[((s * channels + c) * in_h + ih) * in_w + iw]

    _parallel_for[func](batch * channels * kh * kw * out_h * out_w, ctx)


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
    var batch = Int(py=params[13]) if len(params) > 13 else 1
    var ctx = _get_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _im2col[dt](
                out_addr,
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
                channels,
                batch,
                ctx,
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast im2col: " + String(dtype))


# ---------------------------------------------------------------------------
# In-place per-channel bias add on a (batch, channels, plane) tensor:
# out[i] += bias[(i // plane) % channels].
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_chan[
    dtype: DType
](
    out_addr: Int,
    bias_addr: Int,
    total: Int,
    plane: Int,
    channels: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[i] = out_ptr[i] + bias_ptr[(i // plane) % channels]

    _parallel_for[func](total, ctx)


def _bias_add_chan_dispatcher(
    out_buffer: PythonObject,
    bias_buffer: PythonObject,
    params: PythonObject,  # (plane, channels)
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(out_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var bias_addr = Int(py=bias_buffer._data_ptr())
    var total = Int(py=out_buffer.num_elements)
    var plane_val = Int(py=params[0])
    var channels = Int(py=params[1])
    var ctx = _get_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _bias_add_chan[dt](
                out_addr, bias_addr, total, plane_val, channels, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_conv_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("conv_ops")
        b.def_function[_im2col_dispatcher](
            "Im2col",
            docstring="batched NCHW im2col -> (N, C*KH*KW, OH*OW) patch matrix",
        )
        b.def_function[_bias_add_chan_dispatcher](
            "BiasAddChan",
            docstring=(
                "in-place out[i] += bias[(i // plane) % channels] on a"
                " (batch, channels, plane) tensor"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create conv_ops python module: {e}")
