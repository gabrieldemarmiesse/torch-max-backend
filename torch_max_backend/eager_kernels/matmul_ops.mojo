# ===----------------------------------------------------------------------=== #
# Fast eager-mode matmul kernels for max_device, backed by the MAX kernel
# library (`linalg`) — the same kernels the MAX graph compiler uses,
# including the cuBLAS vendor fallback for float dtypes on NVIDIA GPUs.
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

from layout import TileTensor, row_major

from linalg.bmm import batched_matmul
from linalg.matmul.vendor.blas import matmul as vendor_matmul

from op_utils import _get_ctx, _get_dtype, _make_ptr


# ---------------------------------------------------------------------------
# C[m, n] = A[m, k] @ B[k, n]   (or B[n, k] if transpose_b)
# ---------------------------------------------------------------------------


@always_inline
def _matmul[
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
        # Straight to the vendor BLAS library (cuBLAS on NVIDIA) with
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


@always_inline
def _matmul_transb_dispatch[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    if transpose_b != 0:
        _matmul[dtype, True](c_addr, a_addr, b_addr, m, n, k, ctx)
    else:
        _matmul[dtype, False](c_addr, a_addr, b_addr, m, n, k, ctx)


def _matmul_dispatcher(
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
        _matmul_transb_dispatch[DType.float32](
            c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
    elif dtype == DType.float16:
        _matmul_transb_dispatch[DType.float16](
            c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
    elif dtype == DType.bfloat16:
        _matmul_transb_dispatch[DType.bfloat16](
            c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
    else:
        raise Error("unsupported dtype for fast matmul: " + String(dtype))


# ---------------------------------------------------------------------------
# Batched matmul: C[b, m, n] = A[b, m, k] @ B[b, k, n] (or B[b, n, k] if
# transpose_b).
# ---------------------------------------------------------------------------


@always_inline
def _bmm[
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
            batched_matmul[transpose_b=False, target="gpu"](c, a, b, context=ctx)
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _bmm_transb_dispatch[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    if transpose_b != 0:
        _bmm[dtype, True](c_addr, a_addr, b_addr, batch, m, n, k, ctx)
    else:
        _bmm[dtype, False](c_addr, a_addr, b_addr, batch, m, n, k, ctx)


def _bmm_dispatcher(
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
        _bmm_transb_dispatch[DType.float32](
            c_addr, a_addr, b_addr, batch, m, n, k, transpose_b, ctx
        )
    elif dtype == DType.float16:
        _bmm_transb_dispatch[DType.float16](
            c_addr, a_addr, b_addr, batch, m, n, k, transpose_b, ctx
        )
    elif dtype == DType.bfloat16:
        _bmm_transb_dispatch[DType.bfloat16](
            c_addr, a_addr, b_addr, batch, m, n, k, transpose_b, ctx
        )
    else:
        raise Error("unsupported dtype for fast bmm: " + String(dtype))


# ---------------------------------------------------------------------------
# In-place row-broadcast bias add: out[i] += bias[i % cols]. Used as the
# addmm / conv-bias epilogue (the vendor BLAS path can't fuse epilogues).
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_row[
    dtype: DType
](out_addr: Int, bias_addr: Int, total: Int, cols: Int, ctx: DeviceContext) raises:
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

    if dtype == DType.float32:
        _bias_add_row[DType.float32](out_addr, bias_addr, total, cols_val, ctx)
    elif dtype == DType.float16:
        _bias_add_row[DType.float16](out_addr, bias_addr, total, cols_val, ctx)
    elif dtype == DType.bfloat16:
        _bias_add_row[DType.bfloat16](out_addr, bias_addr, total, cols_val, ctx)
    else:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_matmul_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("matmul_ops")
        b.def_function[_matmul_dispatcher](
            "Matmul", docstring="C = A @ B (row-major, optional transposed B), GPU only"
        )
        b.def_function[_bmm_dispatcher](
            "Bmm", docstring="batched C = A @ B (rank 3, optional transposed B), GPU only"
        )
        b.def_function[_bias_add_row_dispatcher](
            "BiasAddRow", docstring="in-place out[i] += bias[i % cols]"
        )
        return b.finalize()
    except e:
        abort(t"failed to create matmul_ops python module: {e}")
