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
from std.math import (
    acos,
    atanh,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    floor,
    log,
    log1p,
    pow,
    sin,
    sinh,
    sqrt,
    tanh,
)
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of, size_of
from std.utils.index import IndexList
from std.utils.coord import Coord
from std.utils.numerics import isnan

from std.algorithm.functional import elementwise

from op_utils import (
    FLOAT_DTYPES,
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
    _scratch_contig,
    _spec_ptr,
    _spec_result,
    _spec_unsupported,
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
#
# Opcodes fall in three buckets:
#   * RELU / ABS / NEG / SIGN work on integer *and* float dtypes and compute
#     directly in the tensor dtype (no float round-trip).
#   * every other opcode is float-only (`_float_unary` below): half-precision
#     inputs are promoted to float32, computed, and cast back — matching
#     torch's numerics and keeping the polynomial math accurate.
# Two of the composed ops deserve a note: `tan` and `asinh` are built from
# sin/cos and log/sqrt rather than the std.math primitives, because those
# lower to libm (`_call_libm`) which `comptime assert`s CPU-only and would
# refuse to compile for the GPU target.
# ---------------------------------------------------------------------------

comptime UOP_RELU = 0
comptime UOP_EXP = 1
comptime UOP_TANH = 2
comptime UOP_ABS = 3
comptime UOP_NEG = 4
comptime UOP_SIGN = 5
comptime UOP_CEIL = 6
comptime UOP_FLOOR = 7
comptime UOP_ACOS = 8
comptime UOP_ASINH = 9
comptime UOP_ATANH = 10
comptime UOP_COS = 11
comptime UOP_COSH = 12
comptime UOP_ERF = 13
comptime UOP_LOG = 14
comptime UOP_LOG1P = 15
comptime UOP_RECIPROCAL = 16
comptime UOP_RSQRT = 17
comptime UOP_SIGMOID = 18
comptime UOP_SILU = 19
comptime UOP_SIN = 20
comptime UOP_SINH = 21
comptime UOP_SQRT = 22
comptime UOP_TAN = 23
comptime UOP_GELU_NONE = 24
comptime UOP_GELU_TANH = 25


@always_inline
def _float_unary[
    dtype: DType, width: Int, op_code: Int
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """The float-only unary math, evaluated in `dtype` (float32 or float64).

    Only instantiated for float32/float64 (half inputs are promoted before
    the call), so every std.math call below sees a supported dtype.
    """
    var res = a
    comptime if op_code == UOP_EXP:
        res = exp(a)
    comptime if op_code == UOP_TANH:
        res = tanh(a)
    comptime if op_code == UOP_CEIL:
        res = ceil(a)
    comptime if op_code == UOP_FLOOR:
        res = floor(a)
    comptime if op_code == UOP_ACOS:
        res = acos(a)
    comptime if op_code == UOP_ASINH:
        # asinh(x) = log(x + sqrt(x^2 + 1)); std.math.asinh is libm/CPU-only.
        res = log(a + sqrt(a * a + 1))
    comptime if op_code == UOP_ATANH:
        res = atanh(a)
    comptime if op_code == UOP_COS:
        res = cos(a)
    comptime if op_code == UOP_COSH:
        res = cosh(a)
    comptime if op_code == UOP_ERF:
        res = erf(a)
    comptime if op_code == UOP_LOG:
        res = log(a)
    comptime if op_code == UOP_LOG1P:
        res = log1p(a)
    comptime if op_code == UOP_RECIPROCAL:
        res = 1 / a
    comptime if op_code == UOP_RSQRT:
        res = 1 / sqrt(a)
    comptime if op_code == UOP_SIGMOID:
        res = 1 / (1 + exp(-a))
    comptime if op_code == UOP_SILU:
        res = a / (1 + exp(-a))
    comptime if op_code == UOP_SIN:
        res = sin(a)
    comptime if op_code == UOP_SINH:
        res = sinh(a)
    comptime if op_code == UOP_SQRT:
        res = sqrt(a)
    comptime if op_code == UOP_TAN:
        # tan(x) = sin(x)/cos(x); std.math.tan is libm/CPU-only.
        res = sin(a) / cos(a)
    comptime if op_code == UOP_GELU_NONE:
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        comptime inv_sqrt2 = 0.70710678118654752440
        res = 0.5 * a * (1 + erf(a * inv_sqrt2))
    comptime if op_code == UOP_GELU_TANH:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        comptime sqrt_2_over_pi = 0.79788456080286535588
        var inner = sqrt_2_over_pi * (a + 0.044715 * a * a * a)
        res = 0.5 * a * (1 + tanh(inner))
    return res


@always_inline
def _unary_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    comptime if not is_direct and not dtype.is_floating_point():
        # Transcendentals / ceil / floor / gelu require a float dtype; the
        # Python side already gates on this, so this only ever fires as a
        # defensive guard (and keeps the float math out of int instantiations).
        raise Error("this unary op requires a floating point dtype")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = in_ptr.load[width=width](i)
            comptime if op_code == UOP_RELU:
                out_ptr.store[width=width](i, max(a, SIMD[dtype, width](0)))
            comptime if op_code == UOP_ABS:
                out_ptr.store[width=width](i, abs(a))
            comptime if op_code == UOP_NEG:
                # `-a` (pop.neg) wraps for unsigned/overflow exactly like torch.
                out_ptr.store[width=width](i, -a)
            comptime if op_code == UOP_SIGN:
                var zero = SIMD[dtype, width](0)
                var pos = a.gt(zero).cast[dtype]()
                var neg = a.lt(zero).cast[dtype]()
                # NaN compares false on both sides -> 0, matching torch.
                out_ptr.store[width=width](i, pos - neg)
            comptime if not is_direct:
                comptime if (dtype == DType.float16 or dtype == DType.bfloat16):
                    var af = a.cast[DType.float32]()
                    out_ptr.store[width=width](
                        i, _float_unary[op_code=op_code](af).cast[dtype]()
                    )
                elif dtype.is_floating_point():
                    out_ptr.store[width=width](
                        i, _float_unary[op_code=op_code](a)
                    )

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
# Unary-to-bool kernels: isnan and logical_not. Input dtype dispatches over
# ints and floats (bool tensors are passed as their uint8 storage); output is
# always bool bytes.
# ---------------------------------------------------------------------------

comptime BUOP_ISNAN = 0
comptime BUOP_LOGICAL_NOT = 1


@always_inline
def _unary_bool[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.load[width=width](i)
        comptime if op_code == BUOP_ISNAN:
            # `numerics.isnan` is bit-based (llvm.is.fpclass), so it survives
            # the fast-math flags that would fold `a != a` to False; it also
            # returns all-False for integer dtypes.
            out_ptr.store[width=width](i, isnan(a))
        comptime if op_code == BUOP_LOGICAL_NOT:
            out_ptr.store[width=width](i, a.eq(SIMD[dtype, width](0)))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("float64 is not supported on GPU")
        else:
            raise Error("no GPU accelerator available at compile time")


def _unary_bool_go[
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
            _unary_bool[dt, op_code](
                _make_ptr[DType.bool](out_addr),
                _make_ptr[dt](in_addr),
                size,
                ctx,
            )
            handled = True
    if not handled:
        abort(String("unsupported dtype for fast unary-to-bool op: ", dtype))


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
# Arange: out[i] = start + i * step, computed in float64 (exact for the
# integer ranges torch produces up to 2^53; the Python caller guards the
# rest) then cast to the buffer dtype. Runs on device so torch.arange on
# max_device never round-trips through a host tensor + blocking H2D copy.
# ---------------------------------------------------------------------------


@always_inline
def _arange[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    start: Float64,
    step: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, start, step)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        out_ptr[i] = (start + Float64(i) * step).cast[dtype]()

    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _arange_go(
    out_ptr: PyObjectPtr,
    start: PyObjectPtr,
    step: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var start_val = _raw_f64(start)
    var step_val = _raw_f64(step)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

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
    ]:
        if dtype == dt:
            _arange[dt](_make_ptr[dt](out_addr), start_val, step_val, size, ctx)
            handled = True
    if not handled:
        abort(String("unsupported dtype for fast arange: ", dtype))


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


def _unary_bool_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _unary_bool_go[op_code](args[0], args[1], args[2], args[3], args[4])
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


def _arange_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        _arange_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): the whole op prologue —
# input checks, output alloc, kernel launch — in one boundary call over
# cached TensorSpecs, reusing the contiguous kernels above. Failed checks
# raise a real NotImplementedError into Python ("take the classic path");
# nothing is swallowed on spec paths.
# ---------------------------------------------------------------------------

# Dtypes the unary spec entries dispatch on for the "direct" (in-dtype) ops;
# the transcendental ops gate down to FLOAT_DTYPES. No float64 (falls back
# to the classic path, which handles it on the CPU device), no bool.
comptime SPEC_UNARY_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]


def _unary_spec_go[op_code: Int](a_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]

    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    var supported = False
    comptime if is_direct:
        comptime for dt in SPEC_UNARY_DTYPES:
            if a.dtype == dt:
                supported = True
    else:
        comptime for dt in FLOAT_DTYPES:
            if a.dtype == dt:
                supported = True
    if not supported:
        raise Error("mojo spec unary: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in SPEC_UNARY_DTYPES:
                if a.dtype == dt:
                    _unary_elementwise[dt, op_code](
                        _make_ptr[dt](addr), _make_ptr[dt](a.ptr), a.numel, ctx
                    )
        else:
            # Mojo-side temporary (design doc §4.7): materialize the strided
            # input into a scratch buffer inside the call — Python never
            # mints a wrapper for it.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in SPEC_UNARY_DTYPES:
                if a.dtype == dt:
                    _unary_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](tmp_addr),
                        a.numel,
                        ctx,
                    )
            _ = tmp^
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


def _unary_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _unary_spec_go[op_code](args[0])
    except e:
        return _spec_unsupported(e)


def _unary_bool_spec_go[op_code: Int](a_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    # bool inputs are read through their uint8 storage (bit-compatible).
    var kdtype = a.dtype
    if a.dtype == DType.bool:
        kdtype = DType.uint8
    var supported = False
    comptime for dt in SPEC_UNARY_DTYPES:
        if kdtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec unary bool: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel  # bool output, itemsize 1
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in SPEC_UNARY_DTYPES:
                if kdtype == dt:
                    _unary_bool[dt, op_code](
                        _make_ptr[DType.bool](addr),
                        _make_ptr[dt](a.ptr),
                        a.numel,
                        ctx,
                    )
        else:
            # Mojo-side temporary; see _unary_spec_go.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in SPEC_UNARY_DTYPES:
                if kdtype == dt:
                    _unary_bool[dt, op_code](
                        _make_ptr[DType.bool](addr),
                        _make_ptr[dt](tmp_addr),
                        a.numel,
                        ctx,
                    )
            _ = tmp^
    return _spec_result(
        buf^, addr, nbytes, a.rank, a.shape, DType.bool, 1, a.numel, a.ctx_ptr
    )


def _unary_bool_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _unary_bool_spec_go[op_code](args[0])
    except e:
        return _spec_unsupported(e)


def _scalar_spec_go[
    op_code: Int
](a_o: PyObjectPtr, scalar_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    if not a.contig:
        raise Error("mojo spec scalar: input not contiguous")
    var supported = False
    comptime for dt in FLOAT_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec scalar: unsupported dtype ", a.dtype)

    var scalar = Float32(_raw_f64(scalar_o))
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        comptime for dt in FLOAT_DTYPES:
            if a.dtype == dt:
                _scalar_elementwise[dt, op_code](
                    _make_ptr[dt](addr),
                    _make_ptr[dt](a.ptr),
                    scalar,
                    a.numel,
                    ctx,
                )
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


def _scalar_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _scalar_spec_go[op_code](args[0], args[1])
    except e:
        return _spec_unsupported(e)


def _int_scalar_spec_go[
    op_code: Int
](a_o: PyObjectPtr, scalar_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    if not a.contig:
        raise Error("mojo spec int scalar: input not contiguous")
    var supported = False
    comptime for dt in [DType.int32, DType.int64]:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec int scalar: unsupported dtype ", a.dtype)

    var scalar = _raw_int(scalar_o)
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        comptime for dt in [DType.int32, DType.int64]:
            if a.dtype == dt:
                _int_scalar_elementwise[dt, op_code](
                    _make_ptr[dt](addr),
                    _make_ptr[dt](a.ptr),
                    scalar,
                    a.numel,
                    ctx,
                )
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


def _int_scalar_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _int_scalar_spec_go[op_code](args[0], args[1])
    except e:
        return _spec_unsupported(e)


comptime SPEC_FILL_DTYPES = [
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
]


def _fill_spec_go(
    shape_t: PyObjectPtr,
    rank_o: PyObjectPtr,
    numel_o: PyObjectPtr,
    value_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_o: PyObjectPtr,
) raises -> PyObjectPtr:
    """Filled-tensor construction: alloc + fill in one boundary call (the
    classic path is a Python-side _alloc plus a Fill call)."""
    var rank = _raw_int(rank_o)
    var numel = _raw_int(numel_o)
    var value = _raw_f64(value_o)
    var dtype = _raw_dtype_int(dtype_o)
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)

    var supported = False
    var itemsize = 0
    comptime for dt in SPEC_FILL_DTYPES:
        if dtype == dt:
            supported = True
            itemsize = size_of[dt]()
    if not supported:
        raise Error("mojo spec fill: unsupported dtype ", dtype)
    # bool fill must store exactly 0/1.
    if dtype == DType.bool:
        value = Float64(1) if value != 0 else Float64(0)

    var ctx = _raw_ctx(ctx_o)
    var nbytes = numel * itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if numel > 0:
        comptime for dt in SPEC_FILL_DTYPES:
            if dtype == dt:
                _fill[dt](_make_ptr[dt](addr), value, numel, ctx)
    return _spec_result(
        buf^, addr, nbytes, rank, shape, dtype, itemsize, numel, _raw_int(ctx_o)
    )


def _fill_spec_dispatcher(
    py_self: PyObjectPtr,
    args: UnsafePointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    try:
        return _fill_spec_go(
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except e:
        return _spec_unsupported(e)



# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_elementwise_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("elementwise_ops")
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RELU],
            "ReluSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); relu",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_EXP],
            "ExpSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); exp",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_TANH],
            "TanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); tanh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ABS],
            "AbsSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); abs",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_NEG],
            "NegSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); neg",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIGN],
            "SignSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sign",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_CEIL],
            "CeilSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); ceil",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_FLOOR],
            "FloorSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); floor",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ACOS],
            "AcosSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); acos",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ASINH],
            "AsinhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); asinh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ATANH],
            "AtanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); atanh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_COS],
            "CosSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); cos",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_COSH],
            "CoshSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); cosh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ERF],
            "ErfSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); erf",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_LOG],
            "LogSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); log",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_LOG1P],
            "Log1pSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); log1p",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RECIPROCAL],
            "ReciprocalSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); reciprocal",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RSQRT],
            "RsqrtSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); rsqrt",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIGMOID],
            "SigmoidSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sigmoid",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SILU],
            "SiluSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); silu",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIN],
            "SinSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sin",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SINH],
            "SinhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sinh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SQRT],
            "SqrtSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sqrt",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_TAN],
            "TanSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); tan",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_GELU_NONE],
            "GeluNoneSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); gelunone",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_GELU_TANH],
            "GeluTanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); gelutanh",
        )
        b.def_py_c_function(
            _unary_bool_spec_dispatcher[BUOP_ISNAN],
            "IsNanSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); isnan -> bool",
        )
        b.def_py_c_function(
            _unary_bool_spec_dispatcher[BUOP_LOGICAL_NOT],
            "LogicalNotSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); logicalnot -> bool",
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_ADD],
            "AddScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_MUL],
            "MulScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_POW],
            "PowScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _int_scalar_spec_dispatcher[IOP_ADD],
            "AddScalarIntSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); int",
        )
        b.def_py_c_function(
            _int_scalar_spec_dispatcher[IOP_MUL],
            "MulScalarIntSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); int",
        )
        b.def_py_c_function(
            _fill_spec_dispatcher,
            "FillSpec",
            docstring="(shape8, rank, numel, value, dtype, ctx) -> result group",
        )
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
            _unary_dispatcher[UOP_ABS],
            "Abs",
            docstring="out = abs(x) (contiguous, int/float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_NEG],
            "Neg",
            docstring="out = -x (contiguous, int/float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SIGN],
            "Sign",
            docstring="out = sign(x) (contiguous, int/float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_CEIL],
            "Ceil",
            docstring="out = ceil(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_FLOOR],
            "Floor",
            docstring="out = floor(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_ACOS],
            "Acos",
            docstring="out = acos(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_ASINH],
            "Asinh",
            docstring="out = asinh(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_ATANH],
            "Atanh",
            docstring="out = atanh(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_COS],
            "Cos",
            docstring="out = cos(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_COSH],
            "Cosh",
            docstring="out = cosh(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_ERF],
            "Erf",
            docstring="out = erf(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_LOG],
            "Log",
            docstring="out = log(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_LOG1P],
            "Log1p",
            docstring="out = log1p(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_RECIPROCAL],
            "Reciprocal",
            docstring="out = 1/x (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_RSQRT],
            "Rsqrt",
            docstring="out = 1/sqrt(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SIGMOID],
            "Sigmoid",
            docstring="out = sigmoid(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SILU],
            "Silu",
            docstring="out = x*sigmoid(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SIN],
            "Sin",
            docstring="out = sin(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SINH],
            "Sinh",
            docstring="out = sinh(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_SQRT],
            "Sqrt",
            docstring="out = sqrt(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_TAN],
            "Tan",
            docstring="out = tan(x) (contiguous, float dtypes)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_GELU_NONE],
            "GeluNone",
            docstring="out = gelu(x) exact erf form (contiguous, float)",
        )
        b.def_py_c_function(
            _unary_dispatcher[UOP_GELU_TANH],
            "GeluTanh",
            docstring="out = gelu(x) tanh approximation (contiguous, float)",
        )
        b.def_py_c_function(
            _unary_bool_dispatcher[BUOP_ISNAN],
            "IsNan",
            docstring="out = (x != x) -> bool (contiguous, int/float)",
        )
        b.def_py_c_function(
            _unary_bool_dispatcher[BUOP_LOGICAL_NOT],
            "LogicalNot",
            docstring="out = (x == 0) -> bool (contiguous, any dtype)",
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
        b.def_py_c_function(
            _arange_dispatcher,
            "Arange",
            docstring="out[i] = start + i * step (contiguous, int/float)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create elementwise_ops python module: {e}")
