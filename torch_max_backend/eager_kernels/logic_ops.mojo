# ===----------------------------------------------------------------------=== #
# Fast eager-mode logic kernels for max_device: broadcast-strided binary
# arithmetic, comparisons (bool output), bitwise ops, bitwise not, and isin.
#
# These cover the small bookkeeping tensors that drive generation loops
# (stopping criteria, attention-mask prep, position ids), where the operands
# frequently broadcast. Each binary kernel takes the contiguous output's
# dims padded to rank 4 plus per-operand strides in elements (0 for
# broadcast dims), computed on the Python side.
#
# Raw-pointer calling convention (mirrors elementwise_ops.mojo /
# tensor_holder.mojo): every Python-visible function receives plain ints —
# each tensor operand as one address int (storage offset already applied),
# dtype as one `max.dtype.DType.value` int per operand role, sizes/counts as
# ints, the dims+strides bundle as a tuple of ints (read with
# `_raw_tuple_int`) — plus the device's DeviceContext pointer as the last
# int. No `max.driver.Buffer` object crosses this boundary anymore.
# Dispatchers are registered as METH_FASTCALL functions
# (`def_py_c_function`) reading raw CPython arguments directly (see
# op_utils), and work is enqueued on MAX's own device queue (fire and
# forget, no sync).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
)


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
# Broadcast-strided binary elementwise: out is contiguous with dims
# d0..d3; each operand is indexed with its own strides (0 on broadcast
# dims). Bitwise ops only instantiate for integer dtypes (the Python side
# routes bool through uint8 views), div only for floats.
# ---------------------------------------------------------------------------

comptime BOP_ADD = 0
comptime BOP_SUB = 1
comptime BOP_MUL = 2
comptime BOP_DIV = 3
comptime BOP_MAX = 4
comptime BOP_MIN = 5
comptime BOP_AND = 6
comptime BOP_OR = 7
comptime BOP_XOR = 8


@always_inline
def _bin_bcast[
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
    comptime if op_code == BOP_DIV and not dtype.is_floating_point():
        raise Error("integer/bool div is not supported in the fast path")
    else:
        comptime if (
            op_code == BOP_AND or op_code == BOP_OR or op_code == BOP_XOR
        ) and dtype.is_floating_point():
            raise Error("bitwise ops require an integer dtype")
        else:
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
                comptime if op_code == BOP_ADD:
                    out_ptr[i] = a + b
                comptime if op_code == BOP_SUB:
                    out_ptr[i] = a - b
                comptime if op_code == BOP_MUL:
                    out_ptr[i] = a * b
                comptime if op_code == BOP_DIV:
                    out_ptr[i] = a / b
                comptime if op_code == BOP_MAX:
                    out_ptr[i] = max(a, b)
                comptime if op_code == BOP_MIN:
                    out_ptr[i] = min(a, b)
                comptime if op_code == BOP_AND:
                    out_ptr[i] = a & b
                comptime if op_code == BOP_OR:
                    out_ptr[i] = a | b
                comptime if op_code == BOP_XOR:
                    out_ptr[i] = a ^ b

            _parallel_for[func](total, ctx)


# ---------------------------------------------------------------------------
# Broadcast-strided comparisons: same indexing, bool output.
# ---------------------------------------------------------------------------

comptime COP_EQ = 0
comptime COP_NE = 1
comptime COP_LT = 2
comptime COP_LE = 3
comptime COP_GT = 4
comptime COP_GE = 5


@always_inline
def _cmp_bcast[
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
    var out_ptr = _make_ptr[DType.bool](out_addr)
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
        comptime if op_code == COP_EQ:
            out_ptr[i] = Scalar[DType.bool](a == b)
        comptime if op_code == COP_NE:
            out_ptr[i] = Scalar[DType.bool](a != b)
        comptime if op_code == COP_LT:
            out_ptr[i] = Scalar[DType.bool](a < b)
        comptime if op_code == COP_LE:
            out_ptr[i] = Scalar[DType.bool](a <= b)
        comptime if op_code == COP_GT:
            out_ptr[i] = Scalar[DType.bool](a > b)
        comptime if op_code == COP_GE:
            out_ptr[i] = Scalar[DType.bool](a >= b)

    _parallel_for[func](total, ctx)


# ---------------------------------------------------------------------------
# Runtime dtype dispatch shared by both broadcast kernel families. The
# strides/dims arrive as one raw CPython 12-tuple (d0..d3, ls0..ls3,
# rs0..rs3), read element-by-element with `_raw_tuple_int` (borrowed
# references, no refcount traffic).
# ---------------------------------------------------------------------------


@always_inline
def _dispatch_bcast[
    kernel_family: Int, op_code: Int
](
    dtype: DType,
    out_addr: Int,
    l_addr: Int,
    r_addr: Int,
    params: PyObjectPtr,
    ctx: DeviceContext,
) raises:
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var ls0 = _raw_tuple_int(params, 4)
    var ls1 = _raw_tuple_int(params, 5)
    var ls2 = _raw_tuple_int(params, 6)
    var ls3 = _raw_tuple_int(params, 7)
    var rs0 = _raw_tuple_int(params, 8)
    var rs1 = _raw_tuple_int(params, 9)
    var rs2 = _raw_tuple_int(params, 10)
    var rs3 = _raw_tuple_int(params, 11)
    var total = d0 * d1 * d2 * d3

    @always_inline
    @parameter
    def run[dt: DType]() raises:
        comptime if kernel_family == 0:
            _bin_bcast[dt, op_code](
                out_addr,
                l_addr,
                r_addr,
                d1,
                d2,
                d3,
                ls0,
                ls1,
                ls2,
                ls3,
                rs0,
                rs1,
                rs2,
                rs3,
                total,
                ctx,
            )
        else:
            _cmp_bcast[dt, op_code](
                out_addr,
                l_addr,
                r_addr,
                d1,
                d2,
                d3,
                ls0,
                ls1,
                ls2,
                ls3,
                rs0,
                rs1,
                rs2,
                rs3,
                total,
                ctx,
            )

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        if dtype == dt:
            run[dt]()
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast broadcast elementwise op: "
            + String(dtype)
        )


def _bin_bcast_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    lhs_ptr: PyObjectPtr,
    rhs_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, ls0..ls3, rs0..rs3)
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    _dispatch_bcast[0, op_code](
        _raw_dtype_int(dtype),
        _raw_int(out_ptr),
        _raw_int(lhs_ptr),
        _raw_int(rhs_ptr),
        params,
        _raw_ctx(ctx_ptr),
    )


def _cmp_bcast_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    lhs_ptr: PyObjectPtr,
    rhs_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, ls0..ls3, rs0..rs3)
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    _dispatch_bcast[1, op_code](
        _raw_dtype_int(dtype),
        _raw_int(out_ptr),
        _raw_int(lhs_ptr),
        _raw_int(rhs_ptr),
        params,
        _raw_ctx(ctx_ptr),
    )


def _bin_bcast_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _bin_bcast_go[op_code](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


def _cmp_bcast_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _cmp_bcast_go[op_code](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Bitwise not over a contiguous span. `~` on bool is logical not, on
# integers the usual complement — matching torch.
# ---------------------------------------------------------------------------


@always_inline
def _bitwise_not[
    dtype: DType
](out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        out_ptr[i] = ~in_ptr[i]

    _parallel_for[func](size, ctx)


def _bitwise_not_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    numel: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var size = _raw_int(numel)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.bool,
        DType.uint8,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
    ]:
        if dtype_val == dt:
            _bitwise_not[dt](out_addr, in_addr, size, ctx)
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast bitwise not: " + String(dtype_val)
        )


def _bitwise_not_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _bitwise_not_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# isin: out[i] = (x[i] in test[0..n_test)) ^ invert. Integer dtypes only
# (float equality-by-value is gated out on the Python side). The inner loop
# over test elements is sequential — n_test is tiny (eos token lists).
# ---------------------------------------------------------------------------


@always_inline
def _isin[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    test_addr: Int,
    size: Int,
    n_test: Int,
    invert: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var test_ptr = _make_ptr[dtype](test_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, test_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var found = False
        for j in range(n_test):
            if in_ptr[i] == test_ptr[j]:
                found = True
                break
        if invert != 0:
            found = not found
        out_ptr[i] = found

    _parallel_for[func](size, ctx)


def _isin_go(
    out_ptr: PyObjectPtr,
    el_ptr: PyObjectPtr,
    te_ptr: PyObjectPtr,
    el_numel: PyObjectPtr,
    te_numel: PyObjectPtr,
    invert: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(el_ptr)
    var test_addr = _raw_int(te_ptr)
    var size = _raw_int(el_numel)
    var n_test_val = _raw_int(te_numel)
    var invert_val = _raw_int(invert)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [DType.int64, DType.int32]:
        if dtype_val == dt:
            _isin[dt](
                out_addr, in_addr, test_addr, size, n_test_val, invert_val, ctx
            )
            handled = True
    if not handled:
        raise Error("unsupported dtype for fast isin: " + String(dtype_val))


def _isin_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _isin_go(
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


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_logic_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("logic_ops")
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_ADD],
            "AddBcast",
            docstring="out = lhs + rhs (broadcast strides)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_SUB],
            "SubBcast",
            docstring="out = lhs - rhs (broadcast strides)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_MUL],
            "MulBcast",
            docstring="out = lhs * rhs (broadcast strides)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_DIV],
            "DivBcast",
            docstring="out = lhs / rhs (broadcast strides, float)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_MAX],
            "MaxBcast",
            docstring="out = max(lhs, rhs) (broadcast strides)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_MIN],
            "MinBcast",
            docstring="out = min(lhs, rhs) (broadcast strides)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_AND],
            "AndBcast",
            docstring="out = lhs & rhs (broadcast strides, int)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_OR],
            "OrBcast",
            docstring="out = lhs | rhs (broadcast strides, int)",
        )
        b.def_py_c_function(
            _bin_bcast_dispatcher[BOP_XOR],
            "XorBcast",
            docstring="out = lhs ^ rhs (broadcast strides, int)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_EQ],
            "EqBcast",
            docstring="out = lhs == rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_NE],
            "NeBcast",
            docstring="out = lhs != rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_LT],
            "LtBcast",
            docstring="out = lhs < rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_LE],
            "LeBcast",
            docstring="out = lhs <= rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_GT],
            "GtBcast",
            docstring="out = lhs > rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _cmp_bcast_dispatcher[COP_GE],
            "GeBcast",
            docstring="out = lhs >= rhs -> bool (broadcast strides)",
        )
        b.def_py_c_function(
            _bitwise_not_dispatcher,
            "BitwiseNot",
            docstring="out = ~x (bool/int, contiguous)",
        )
        b.def_py_c_function(
            _isin_dispatcher,
            "IsIn",
            docstring="out[i] = x[i] in test (int dtypes) ^ invert",
        )
        return b.finalize()
    except e:
        abort(t"failed to create logic_ops python module: {e}")
