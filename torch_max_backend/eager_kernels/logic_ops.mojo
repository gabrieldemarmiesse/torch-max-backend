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
# strides/dims arrive as one 12-tuple (d0..d3, ls0..ls3, rs0..rs3).
# ---------------------------------------------------------------------------


@always_inline
def _dispatch_bcast[
    kernel_family: Int, op_code: Int
](
    dtype: DType,
    out_addr: Int,
    l_addr: Int,
    r_addr: Int,
    params: PythonObject,
    ctx: DeviceContext,
) raises:
    var d0 = Int(py=params[0])
    var d1 = Int(py=params[1])
    var d2 = Int(py=params[2])
    var d3 = Int(py=params[3])
    var ls0 = Int(py=params[4])
    var ls1 = Int(py=params[5])
    var ls2 = Int(py=params[6])
    var ls3 = Int(py=params[7])
    var rs0 = Int(py=params[8])
    var rs1 = Int(py=params[9])
    var rs2 = Int(py=params[10])
    var rs3 = Int(py=params[11])
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

    if dtype == DType.float32:
        run[DType.float32]()
    elif dtype == DType.float16:
        run[DType.float16]()
    elif dtype == DType.bfloat16:
        run[DType.bfloat16]()
    elif dtype == DType.int8:
        run[DType.int8]()
    elif dtype == DType.int16:
        run[DType.int16]()
    elif dtype == DType.int32:
        run[DType.int32]()
    elif dtype == DType.int64:
        run[DType.int64]()
    elif dtype == DType.uint8:
        run[DType.uint8]()
    else:
        raise Error(
            "unsupported dtype for fast broadcast elementwise op: "
            + String(dtype)
        )


def _bin_bcast_dispatcher[
    op_code: Int
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    params: PythonObject,  # (d0..d3, ls0..ls3, rs0..rs3)
    device_context_ptr: PythonObject,
) raises:
    _dispatch_bcast[0, op_code](
        _get_dtype(lhs_buffer),
        Int(py=out_buffer._data_ptr()),
        Int(py=lhs_buffer._data_ptr()),
        Int(py=rhs_buffer._data_ptr()),
        params,
        _get_ctx(device_context_ptr),
    )


def _cmp_bcast_dispatcher[
    op_code: Int
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    params: PythonObject,  # (d0..d3, ls0..ls3, rs0..rs3)
    device_context_ptr: PythonObject,
) raises:
    _dispatch_bcast[1, op_code](
        _get_dtype(lhs_buffer),
        Int(py=out_buffer._data_ptr()),
        Int(py=lhs_buffer._data_ptr()),
        Int(py=rhs_buffer._data_ptr()),
        params,
        _get_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# Bitwise not over a contiguous buffer. `~` on bool is logical not, on
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


def _bitwise_not_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var size = Int(py=out_buffer.num_elements)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.bool:
        _bitwise_not[DType.bool](out_addr, in_addr, size, ctx)
    elif dtype == DType.uint8:
        _bitwise_not[DType.uint8](out_addr, in_addr, size, ctx)
    elif dtype == DType.int8:
        _bitwise_not[DType.int8](out_addr, in_addr, size, ctx)
    elif dtype == DType.int16:
        _bitwise_not[DType.int16](out_addr, in_addr, size, ctx)
    elif dtype == DType.int32:
        _bitwise_not[DType.int32](out_addr, in_addr, size, ctx)
    elif dtype == DType.int64:
        _bitwise_not[DType.int64](out_addr, in_addr, size, ctx)
    else:
        raise Error("unsupported dtype for fast bitwise not: " + String(dtype))


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


def _isin_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    test_buffer: PythonObject,
    n_test: PythonObject,
    invert: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var out_addr = Int(py=out_buffer._data_ptr())
    var in_addr = Int(py=in_buffer._data_ptr())
    var test_addr = Int(py=test_buffer._data_ptr())
    var size = Int(py=out_buffer.num_elements)
    var n_test_val = Int(py=n_test)
    var invert_val = Int(py=invert)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.int64:
        _isin[DType.int64](
            out_addr, in_addr, test_addr, size, n_test_val, invert_val, ctx
        )
    elif dtype == DType.int32:
        _isin[DType.int32](
            out_addr, in_addr, test_addr, size, n_test_val, invert_val, ctx
        )
    else:
        raise Error("unsupported dtype for fast isin: " + String(dtype))


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_logic_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("logic_ops")
        b.def_function[_bin_bcast_dispatcher[BOP_ADD]](
            "AddBcast", docstring="out = lhs + rhs (broadcast strides)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_SUB]](
            "SubBcast", docstring="out = lhs - rhs (broadcast strides)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_MUL]](
            "MulBcast", docstring="out = lhs * rhs (broadcast strides)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_DIV]](
            "DivBcast", docstring="out = lhs / rhs (broadcast strides, float)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_MAX]](
            "MaxBcast", docstring="out = max(lhs, rhs) (broadcast strides)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_MIN]](
            "MinBcast", docstring="out = min(lhs, rhs) (broadcast strides)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_AND]](
            "AndBcast", docstring="out = lhs & rhs (broadcast strides, int)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_OR]](
            "OrBcast", docstring="out = lhs | rhs (broadcast strides, int)"
        )
        b.def_function[_bin_bcast_dispatcher[BOP_XOR]](
            "XorBcast", docstring="out = lhs ^ rhs (broadcast strides, int)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_EQ]](
            "EqBcast", docstring="out = lhs == rhs -> bool (broadcast strides)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_NE]](
            "NeBcast", docstring="out = lhs != rhs -> bool (broadcast strides)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_LT]](
            "LtBcast", docstring="out = lhs < rhs -> bool (broadcast strides)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_LE]](
            "LeBcast", docstring="out = lhs <= rhs -> bool (broadcast strides)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_GT]](
            "GtBcast", docstring="out = lhs > rhs -> bool (broadcast strides)"
        )
        b.def_function[_cmp_bcast_dispatcher[COP_GE]](
            "GeBcast", docstring="out = lhs >= rhs -> bool (broadcast strides)"
        )
        b.def_function[_bitwise_not_dispatcher](
            "BitwiseNot", docstring="out = ~x (bool/int, contiguous)"
        )
        b.def_function[_isin_dispatcher](
            "IsIn", docstring="out[i] = x[i] in test (int dtypes) ^ invert"
        )
        return b.finalize()
    except e:
        abort(t"failed to create logic_ops python module: {e}")
