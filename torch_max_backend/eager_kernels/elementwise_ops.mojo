# ===----------------------------------------------------------------------=== #
# Fast eager-mode elementwise kernels for max_device.
#
# This module is imported from Python through `mojo.importer`: the first
# import runs `mojo build --emit shared-lib` and caches the resulting
# CPython extension under `__mojocache__/` next to this file (content
# addressed, so editing this file triggers exactly one recompile).
#
# The design mirrors `max._interpreter_ops.elementwise_binary_ops` (the MO
# interpreter's own op bindings): each Python-visible function receives raw
# tensor-data pointers (plain ints, storage offset already applied) plus
# explicit numel/dtype ints and the device's DeviceContext pointer — there
# are no `max.driver.Buffer` objects and no attribute access at all —
# dispatches on dtype at *runtime* (all dtype specializations are compiled
# into this one extension), and enqueues the kernel on MAX's own device
# context — so ordering with regular MAX driver operations (copies, other
# kernels) comes for free.
#
# Every kernel here works on *contiguous* buffers with fully dynamic sizes:
# one compiled extension serves every shape with zero recompilation.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.math import exp, pow, tanh
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

from op_utils import (
    FLOAT_DTYPES,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
)

# ---------------------------------------------------------------------------
# Raw-pointer calling convention: every Python-visible kernel below receives
# tensor operands as a single int (the `._ptr` address, storage offset
# already applied), unpacked with `_raw_int` and turned into a typed
# pointer with `_make_ptr[dt]`; numel and dtype are explicit int args
# (`_raw_int` / `_raw_dtype_int`); `ctx_ptr` (int) is always last
# (`_raw_ctx`). The dispatchers register as METH_FASTCALL functions
# (`def_py_c_function`), skipping the owning PythonObject wrappers of the
# `def_function` path entirely.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Binary elementwise kernels
# ---------------------------------------------------------------------------

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAX = 4
comptime OP_MIN = 5


@always_inline
def _bin_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """out = op(lhs, rhs) over `size` contiguous elements."""

    comptime if op_code == OP_DIV and not dtype.is_floating_point():
        raise Error("integer/bool div is not supported in the fast path")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, lhs_ptr, rhs_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = lhs_ptr.load[width=width](i)
            var b = rhs_ptr.load[width=width](i)
            comptime if op_code == OP_ADD:
                out_ptr.store[width=width](i, a + b)
            comptime if op_code == OP_SUB:
                out_ptr.store[width=width](i, a - b)
            comptime if op_code == OP_MUL:
                out_ptr.store[width=width](i, a * b)
            comptime if op_code == OP_DIV:
                out_ptr.store[width=width](i, a / b)
            comptime if op_code == OP_MAX:
                out_ptr.store[width=width](i, max(a, b))
            comptime if op_code == OP_MIN:
                out_ptr.store[width=width](i, min(a, b))

        if ctx.api() == "cpu":
            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                comptime if dtype != DType.float64:
                    elementwise[func, simd_width=1, target="gpu"](
                        Coord(size), ctx
                    )
                else:
                    raise Error("float64 is not supported on GPU")
            else:
                raise Error("no GPU accelerator available at compile time")


def _bin_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    lhs_ptr: PyObjectPtr,
    rhs_ptr: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var lhs_addr = _raw_int(lhs_ptr)
    var rhs_addr = _raw_int(rhs_ptr)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        if dtype == dt:
            _bin_elementwise[dt, op_code](
                _make_ptr[dt](out_addr),
                _make_ptr[dt](lhs_addr),
                _make_ptr[dt](rhs_addr),
                size,
                ctx,
            )
            handled = True
    if not handled:
        abort(
            String("unsupported dtype for fast binary elementwise op: ", dtype)
        )


# ---------------------------------------------------------------------------
# Unary elementwise kernels
# ---------------------------------------------------------------------------

comptime UOP_RELU = 0
comptime UOP_EXP = 1
comptime UOP_TANH = 2


@always_inline
def _unary_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime if (
        op_code == UOP_EXP or op_code == UOP_TANH
    ) and not dtype.is_floating_point():
        raise Error("exp/tanh require a floating point dtype")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = in_ptr.load[width=width](i)
            comptime if op_code == UOP_RELU:
                out_ptr.store[width=width](i, max(a, SIMD[dtype, width](0)))
            comptime if op_code == UOP_EXP:
                # The inner comptime gate gives the constraint checker
                # direct evidence that `exp` only instantiates for floats;
                # the runtime raise above already guarantees we never get
                # here otherwise. Half-precision inputs are computed in
                # float32 to match torch's numerics.
                comptime if dtype == DType.float16 or dtype == DType.bfloat16:
                    out_ptr.store[width=width](
                        i, exp(a.cast[DType.float32]()).cast[dtype]()
                    )
                elif dtype.is_floating_point():
                    out_ptr.store[width=width](i, exp(a))
            comptime if op_code == UOP_TANH:
                comptime if dtype == DType.float16 or dtype == DType.bfloat16:
                    out_ptr.store[width=width](
                        i, tanh(a.cast[DType.float32]()).cast[dtype]()
                    )
                elif dtype.is_floating_point():
                    out_ptr.store[width=width](i, tanh(a))

        if ctx.api() == "cpu":
            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                comptime if dtype != DType.float64:
                    elementwise[func, simd_width=1, target="gpu"](
                        Coord(size), ctx
                    )
                else:
                    raise Error("float64 is not supported on GPU")
            else:
                raise Error("no GPU accelerator available at compile time")


def _unary_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _unary_elementwise[dt, op_code](
                _make_ptr[dt](out_addr),
                _make_ptr[dt](in_addr),
                size,
                ctx,
            )
            handled = True
    if not handled:
        abort(
            String("unsupported dtype for fast unary elementwise op: ", dtype)
        )


# ---------------------------------------------------------------------------
# Scalar-operand elementwise kernels: out = op(x, scalar). The scalar comes
# in as a Python float and is applied in float32 (cast back to the tensor
# dtype), matching torch's promotion for float tensors. Float dtypes only —
# the Python side gates on that.
# ---------------------------------------------------------------------------

comptime SOP_ADD = 0
comptime SOP_MUL = 1
comptime SOP_POW = 2


@always_inline
def _scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Float32,
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime if not dtype.is_floating_point():
        raise Error("scalar elementwise ops require a floating point dtype")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr, scalar)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = in_ptr.load[width=width](i).cast[DType.float32]()
            var s = SIMD[DType.float32, width](scalar)
            comptime if op_code == SOP_ADD:
                out_ptr.store[width=width](i, (a + s).cast[dtype]())
            comptime if op_code == SOP_MUL:
                out_ptr.store[width=width](i, (a * s).cast[dtype]())
            comptime if op_code == SOP_POW:
                out_ptr.store[width=width](i, pow(a, s).cast[dtype]())

        if ctx.api() == "cpu":
            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("no GPU accelerator available at compile time")


def _scalar_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    scalar: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var scalar_val = Float32(_raw_f64(scalar))
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            _scalar_elementwise[dt, op_code](
                _make_ptr[dt](out_addr),
                _make_ptr[dt](in_addr),
                scalar_val,
                size,
                ctx,
            )
            handled = True
    if not handled:
        abort(
            String("unsupported dtype for fast scalar elementwise op: ", dtype)
        )


# Integer variant: out = op(x, scalar) over integer dtypes (int semantics,
# no float round-trip).

comptime IOP_ADD = 0
comptime IOP_MUL = 1


@always_inline
def _int_scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Int,
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, scalar)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.load[width=width](i)
        comptime if op_code == IOP_ADD:
            out_ptr.store[width=width](i, a + SIMD[dtype, width](scalar))
        comptime if op_code == IOP_MUL:
            out_ptr.store[width=width](i, a * SIMD[dtype, width](scalar))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _int_scalar_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    scalar: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var scalar_val = _raw_int(scalar)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [DType.int64, DType.int32]:
        if dtype == dt:
            _int_scalar_elementwise[dt, op_code](
                _make_ptr[dt](out_addr),
                _make_ptr[dt](in_addr),
                scalar_val,
                size,
                ctx,
            )
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast int scalar op: " + String(dtype)
        )


# ---------------------------------------------------------------------------
# Fill: out[i] = value, over any dtype. The value comes in as a Float64 and
# is cast to the buffer dtype (exact for the small ints masks/one-hots use).
# ---------------------------------------------------------------------------


@always_inline
def _fill[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    value: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    var scalar = value.cast[dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr, scalar)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        out_ptr.store[width=width](i, SIMD[dtype, width](scalar))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _fill_go(
    out_ptr: PyObjectPtr,
    value: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var value_val = _raw_f64(value)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    # bool fill must store exactly 0/1 (a raw nonzero float doesn't reliably
    # cast to True); normalize once so every dtype's call site is identical.
    if dtype == DType.bool:
        value_val = Float64(1) if value_val != 0 else Float64(0)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
        DType.bool,
    ]:
        if dtype == dt:
            _fill[dt](_make_ptr[dt](out_addr), value_val, size, ctx)
            handled = True
    if not handled:
        abort(String("unsupported dtype for fast fill: ", dtype))


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers in aten_fast.py; errors cannot cross the C ABI, and the
# only raise sites are unsupported-dtype guards already gated upstream.
# ---------------------------------------------------------------------------


def _bin_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _bin_go[op_code](args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _unary_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _unary_go[op_code](args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


def _scalar_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _scalar_go[op_code](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


def _int_scalar_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _int_scalar_go[op_code](
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except:
        pass
    return _raw_ret_none()


def _fill_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _fill_go(args[0], args[1], args[2], args[3], args[4])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_elementwise_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("elementwise_ops")
        b.def_py_c_function(
            _bin_dispatcher[OP_ADD],
            "Add",
            docstring="out = lhs + rhs (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_SUB],
            "Sub",
            docstring="out = lhs - rhs (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_MUL],
            "Mul",
            docstring="out = lhs * rhs (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_DIV],
            "Div",
            docstring="out = lhs / rhs (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_MAX],
            "Max",
            docstring="out = max(lhs, rhs) (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_MIN],
            "Min",
            docstring="out = min(lhs, rhs) (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_RELU],
            "Relu",
            docstring="out = max(x, 0) (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_EXP],
            "Exp",
            docstring="out = exp(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_TANH],
            "Tanh",
            docstring="out = tanh(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _scalar_dispatcher[SOP_ADD],
            "AddScalar",
            docstring="out = x + scalar (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _scalar_dispatcher[SOP_MUL],
            "MulScalar",
            docstring="out = x * scalar (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _scalar_dispatcher[SOP_POW],
            "PowScalar",
            docstring="out = x ** scalar (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _int_scalar_dispatcher[IOP_ADD],
            "AddScalarInt",
            docstring="out = x + scalar (contiguous, int dtypes)",
        )
        b.def_py_c_function(
            _int_scalar_dispatcher[IOP_MUL],
            "MulScalarInt",
            docstring="out = x * scalar (contiguous, int dtypes)",
        )
        b.def_py_c_function(
            _fill_dispatcher,
            "Fill",
            docstring="out[i] = value (contiguous, any dtype)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create elementwise_ops python module: {e}")
