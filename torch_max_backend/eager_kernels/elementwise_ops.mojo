# ===----------------------------------------------------------------------=== #
# Fast eager-mode elementwise kernels for max_device.
#
# This module is imported from Python through `mojo.importer`: the first
# import runs `mojo build --emit shared-lib` and caches the resulting
# CPython extension under `__mojocache__/` next to this file (content
# addressed, so editing this file triggers exactly one recompile).
#
# The design mirrors `max._interpreter_ops.elementwise_binary_ops` (the MO
# interpreter's own op bindings): each Python-visible function receives the
# `max.driver.Buffer` objects directly plus the device's DeviceContext
# pointer, unwraps raw pointers, dispatches on dtype at *runtime* (all dtype
# specializations are compiled into this one extension), and enqueues the
# kernel on MAX's own device context — so ordering with regular MAX driver
# operations (copies, other kernels) comes for free.
#
# Every kernel here works on *contiguous* buffers with fully dynamic sizes:
# one compiled extension serves every shape with zero recompilation.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu.host import DeviceContext
from std.math import exp, pow, tanh
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

# ---------------------------------------------------------------------------
# Helpers to unwrap `max.driver.Buffer` Python objects.
# Mirrors `max._interpreter_ops.op_utils`.
# ---------------------------------------------------------------------------


def _get_dtype(buffer: PythonObject) raises -> DType:
    return DType._from_ui8(UInt8(py=buffer.dtype.value)._mlir_value)


def _get_buffer_ptr[
    dtype: DType
](buffer: PythonObject) raises -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    return UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(py=buffer._data_ptr())
    )


def _get_size(buffer: PythonObject) raises -> Int:
    return Int(py=buffer.num_elements)


def _get_ctx(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=addr)
    )


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


def _bin_dispatcher[
    op_code: Int
](
    out_buffer: PythonObject,
    lhs_buffer: PythonObject,
    rhs_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Runtime dtype dispatch for a binary elementwise op.

    All dtype specializations live in this one extension, so a single
    `.so` serves every dtype and every shape.
    """
    var dtype = _get_dtype(lhs_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _bin_elementwise[DType.float32, op_code](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](lhs_buffer),
            _get_buffer_ptr[DType.float32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float16:
        _bin_elementwise[DType.float16, op_code](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](lhs_buffer),
            _get_buffer_ptr[DType.float16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        _bin_elementwise[DType.bfloat16, op_code](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](lhs_buffer),
            _get_buffer_ptr[DType.bfloat16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        _bin_elementwise[DType.float64, op_code](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.float64](lhs_buffer),
            _get_buffer_ptr[DType.float64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        _bin_elementwise[DType.int8, op_code](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.int8](lhs_buffer),
            _get_buffer_ptr[DType.int8](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        _bin_elementwise[DType.int16, op_code](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.int16](lhs_buffer),
            _get_buffer_ptr[DType.int16](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        _bin_elementwise[DType.int32, op_code](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](lhs_buffer),
            _get_buffer_ptr[DType.int32](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        _bin_elementwise[DType.int64, op_code](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](lhs_buffer),
            _get_buffer_ptr[DType.int64](rhs_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        _bin_elementwise[DType.uint8, op_code](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.uint8](lhs_buffer),
            _get_buffer_ptr[DType.uint8](rhs_buffer),
            size,
            ctx,
        )
    else:
        raise Error(
            "unsupported dtype for fast binary elementwise op: " + String(dtype)
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


def _unary_dispatcher[
    op_code: Int
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _unary_elementwise[DType.float32, op_code](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float16:
        _unary_elementwise[DType.float16, op_code](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        _unary_elementwise[DType.bfloat16, op_code](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            size,
            ctx,
        )
    else:
        raise Error(
            "unsupported dtype for fast unary elementwise op: " + String(dtype)
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


def _scalar_dispatcher[
    op_code: Int
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    scalar: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var size = _get_size(out_buffer)
    var scalar_val = Float32(py=scalar)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float32:
        _scalar_elementwise[DType.float32, op_code](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.float32](in_buffer),
            scalar_val,
            size,
            ctx,
        )
    elif dtype == DType.float16:
        _scalar_elementwise[DType.float16, op_code](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.float16](in_buffer),
            scalar_val,
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        _scalar_elementwise[DType.bfloat16, op_code](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bfloat16](in_buffer),
            scalar_val,
            size,
            ctx,
        )
    else:
        raise Error(
            "unsupported dtype for fast scalar elementwise op: " + String(dtype)
        )


# Integer variant: out = x + scalar over integer dtypes (int semantics, no
# float round-trip). Only Add is needed so far.


@always_inline
def _add_scalar_int[
    dtype: DType
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
        out_ptr.store[width=width](i, a + SIMD[dtype, width](scalar))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _add_scalar_int_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    scalar: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var dtype = _get_dtype(in_buffer)
    var size = _get_size(out_buffer)
    var scalar_val = Int(py=scalar)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.int64:
        _add_scalar_int[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.int64](in_buffer),
            scalar_val,
            size,
            ctx,
        )
    elif dtype == DType.int32:
        _add_scalar_int[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.int32](in_buffer),
            scalar_val,
            size,
            ctx,
        )
    else:
        raise Error(
            "unsupported dtype for fast int scalar add: " + String(dtype)
        )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_elementwise_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("elementwise_ops")
        b.def_function[_bin_dispatcher[OP_ADD]](
            "Add", docstring="out = lhs + rhs (contiguous, dtype dispatch)"
        )
        b.def_function[_bin_dispatcher[OP_SUB]](
            "Sub", docstring="out = lhs - rhs (contiguous, dtype dispatch)"
        )
        b.def_function[_bin_dispatcher[OP_MUL]](
            "Mul", docstring="out = lhs * rhs (contiguous, dtype dispatch)"
        )
        b.def_function[_bin_dispatcher[OP_DIV]](
            "Div", docstring="out = lhs / rhs (contiguous, dtype dispatch)"
        )
        b.def_function[_bin_dispatcher[OP_MAX]](
            "Max", docstring="out = max(lhs, rhs) (contiguous, dtype dispatch)"
        )
        b.def_function[_bin_dispatcher[OP_MIN]](
            "Min", docstring="out = min(lhs, rhs) (contiguous, dtype dispatch)"
        )
        b.def_function[_unary_dispatcher[UOP_RELU]](
            "Relu", docstring="out = max(x, 0) (contiguous, dtype dispatch)"
        )
        b.def_function[_unary_dispatcher[UOP_EXP]](
            "Exp", docstring="out = exp(x) (contiguous, float dtypes)"
        )
        b.def_function[_unary_dispatcher[UOP_TANH]](
            "Tanh", docstring="out = tanh(x) (contiguous, float dtypes)"
        )
        b.def_function[_scalar_dispatcher[SOP_ADD]](
            "AddScalar", docstring="out = x + scalar (contiguous, float dtypes)"
        )
        b.def_function[_scalar_dispatcher[SOP_MUL]](
            "MulScalar", docstring="out = x * scalar (contiguous, float dtypes)"
        )
        b.def_function[_scalar_dispatcher[SOP_POW]](
            "PowScalar",
            docstring="out = x ** scalar (contiguous, float dtypes)",
        )
        b.def_function[_add_scalar_int_dispatcher](
            "AddScalarInt",
            docstring="out = x + scalar (contiguous, int dtypes)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create elementwise_ops python module: {e}")
