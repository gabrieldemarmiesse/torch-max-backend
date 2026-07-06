# ===----------------------------------------------------------------------=== #
# Fast eager-mode NN kernels for max_device: batch norm (inference),
# layer norm, row softmax (with optional causal mask), spatial mean,
# max pool (with indices), embedding gather, and boolean all-reduce.
#
# Same architecture as elementwise_ops.mojo: Python-visible functions get
# `max.driver.Buffer` objects plus the device's DeviceContext pointer, and
# enqueue work on MAX's own device queue (fire and forget, no sync).
#
# All kernels here are written as a parallel-for over independent output
# elements or rows (`elementwise` with an inner sequential loop), so the same
# code runs on CPU and GPU with fully dynamic shapes.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.math import sqrt, exp
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord
from std.utils.numerics import min_or_neg_inf

from std.algorithm.functional import elementwise

from op_utils import _get_ctx, _get_dtype, _make_ptr


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


def _batch_norm_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    mean_buffer: PythonObject,
    var_buffer: PythonObject,
    gamma_buffer: PythonObject,
    beta_buffer: PythonObject,
    params: PythonObject,  # (eps, channels, inner)
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var mean_addr = Int(py=mean_buffer._data_ptr())
    var var_addr = Int(py=var_buffer._data_ptr())
    var gamma_addr = Int(py=gamma_buffer._data_ptr())
    var beta_addr = Int(py=beta_buffer._data_ptr())
    var eps_val = Float32(py=params[0])
    var channels_val = Int(py=params[1])
    var inner_val = Int(py=params[2])
    var total = Int(py=out_buffer.num_elements)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _batch_norm[DType.float32](
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
    elif dtype == DType.float16:
        _batch_norm[DType.float16](
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
    elif dtype == DType.bfloat16:
        _batch_norm[DType.bfloat16](
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
    else:
        raise Error("unsupported dtype for fast batch_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Layer norm over the last dim. One parallel task per row; also writes the
# per-row mean and rstd (float32), matching aten.native_layer_norm outputs.
# ---------------------------------------------------------------------------


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


def _layer_norm_dispatcher(
    out_buffer: PythonObject,
    mean_out_buffer: PythonObject,
    rstd_out_buffer: PythonObject,
    in_buffer: PythonObject,
    gamma_buffer: PythonObject,
    beta_buffer: PythonObject,
    params: PythonObject,  # (eps, rows, cols)
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var mean_out_addr = Int(py=mean_out_buffer._data_ptr())
    var rstd_out_addr = Int(py=rstd_out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var gamma_addr = Int(py=gamma_buffer._data_ptr())
    var beta_addr = Int(py=beta_buffer._data_ptr())
    var eps_val = Float32(py=params[0])
    var rows_val = Int(py=params[1])
    var cols_val = Int(py=params[2])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _layer_norm[DType.float32](
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
    elif dtype == DType.float16:
        _layer_norm[DType.float16](
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
    elif dtype == DType.bfloat16:
        _layer_norm[DType.bfloat16](
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
    else:
        raise Error("unsupported dtype for fast layer_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise softmax with optional scaling and causal masking, for attention.
# Input is (rows, cols) where rows = batch * q_len. With causal=1, row r
# (query index r % q_len) only attends to columns j <= r % q_len — the
# top-left-aligned tril(ones(L, S)) mask that torch's sdpa is_causal=True
# specifies; masked columns get probability 0.
# ---------------------------------------------------------------------------


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


def _softmax_rows_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    scale: PythonObject,
    causal: PythonObject,
    q_len: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var rows_val = Int(py=rows)
    var cols_val = Int(py=cols)
    var scale_val = Float32(py=scale)
    var causal_val = Int(py=causal)
    var q_len_val = Int(py=q_len)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _softmax_rows[DType.float32](
            out_addr,
            in_addr,
            rows_val,
            cols_val,
            scale_val,
            causal_val,
            q_len_val,
            ctx,
        )
    elif dtype == DType.float16:
        _softmax_rows[DType.float16](
            out_addr,
            in_addr,
            rows_val,
            cols_val,
            scale_val,
            causal_val,
            q_len_val,
            ctx,
        )
    elif dtype == DType.bfloat16:
        _softmax_rows[DType.bfloat16](
            out_addr,
            in_addr,
            rows_val,
            cols_val,
            scale_val,
            causal_val,
            q_len_val,
            ctx,
        )
    else:
        raise Error("unsupported dtype for fast softmax: " + String(dtype))


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


def _mean_rows_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var rows_val = Int(py=rows)
    var cols_val = Int(py=cols)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _mean_rows[DType.float32](out_addr, in_addr, rows_val, cols_val, ctx)
    elif dtype == DType.float16:
        _mean_rows[DType.float16](out_addr, in_addr, rows_val, cols_val, ctx)
    elif dtype == DType.bfloat16:
        _mean_rows[DType.bfloat16](out_addr, in_addr, rows_val, cols_val, ctx)
    else:
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


def _max_pool2d_dispatcher(
    out_buffer: PythonObject,
    idx_buffer: PythonObject,
    in_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var idx_addr = Int(py=idx_buffer._data_ptr())
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
    var planes = Int(py=params[12])
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _max_pool2d[DType.float32](
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
    elif dtype == DType.float16:
        _max_pool2d[DType.float16](
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
    elif dtype == DType.bfloat16:
        _max_pool2d[DType.bfloat16](
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
    else:
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
    if dtype == DType.float32:
        _gather0[DType.float32, idx_dtype](
            out_addr, weight_addr, indices_addr, num_indices, row_len, ctx
        )
    elif dtype == DType.float16:
        _gather0[DType.float16, idx_dtype](
            out_addr, weight_addr, indices_addr, num_indices, row_len, ctx
        )
    elif dtype == DType.bfloat16:
        _gather0[DType.bfloat16, idx_dtype](
            out_addr, weight_addr, indices_addr, num_indices, row_len, ctx
        )
    else:
        raise Error("unsupported dtype for fast embedding: " + String(dtype))


def _gather0_dispatcher(
    out_buffer: PythonObject,
    weight_buffer: PythonObject,
    indices_buffer: PythonObject,
    num_indices: PythonObject,
    row_len: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(weight_buffer)
    var idx_dtype = _get_dtype(indices_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var weight_addr = Int(py=weight_buffer._data_ptr())
    var indices_addr = Int(py=indices_buffer._data_ptr())
    var num_indices_val = Int(py=num_indices)
    var row_len_val = Int(py=row_len)
    var ctx = _get_ctx(device_context_ptr)

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
# all() over a bool tensor -> scalar bool. Single sequential task; only used
# for small tensors (the Python side gates on size).
# ---------------------------------------------------------------------------


@always_inline
def _all_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[DType.bool](in_addr)

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


def _all_bool_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    size: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    _all_bool(
        Int(py=out_buffer._data_ptr()),
        Int(py=in_buffer._data_ptr()),
        Int(py=size),
        _get_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_nn_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("nn_ops")
        b.def_function[_batch_norm_dispatcher](
            "BatchNormInference",
            docstring=(
                "out = (x - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]"
                " (NC..., contiguous)"
            ),
        )
        b.def_function[_layer_norm_dispatcher](
            "LayerNorm",
            docstring=(
                "layer norm over the last dim; also writes float32 mean/rstd"
                " per row"
            ),
        )
        b.def_function[_softmax_rows_dispatcher](
            "SoftmaxRows",
            docstring="row softmax of scale*x with optional causal mask",
        )
        b.def_function[_mean_rows_dispatcher](
            "MeanRows",
            docstring="mean over the trailing dims (rows, cols) -> (rows,)",
        )
        b.def_function[_max_pool2d_dispatcher](
            "MaxPool2dWithIndices",
            docstring=(
                "max pool over NCHW contiguous input, returns values and int64"
                " plane indices"
            ),
        )
        b.def_function[_gather0_dispatcher](
            "Gather0", docstring="embedding lookup: gather rows of a 2D table"
        )
        b.def_function[_all_bool_dispatcher](
            "AllBool", docstring="all() over a bool tensor -> scalar bool"
        )
        return b.finalize()
    except e:
        abort(t"failed to create nn_ops python module: {e}")
